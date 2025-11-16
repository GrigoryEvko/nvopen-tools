// Function: sub_BD60C0
// Address: 0xbd60c0
//
__int64 __fastcall sub_BD60C0(_QWORD *a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // r12
  unsigned __int64 v4; // rcx
  __int64 v5; // rsi
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // r8d
  __int64 *v9; // rdx
  __int64 v10; // r9
  int v11; // edx
  int v12; // r10d

  result = a1[1];
  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v2 = result;
  if ( result )
  {
    *(_QWORD *)result = *(_QWORD *)result & 7LL | v2;
  }
  else
  {
    result = *(_QWORD *)sub_BD5C60(a1[2]);
    v4 = *(_QWORD *)(result + 3176);
    if ( v2 >= v4 )
    {
      v5 = *(unsigned int *)(result + 3192);
      if ( v2 < v4 + 16 * v5 )
      {
        v6 = a1[2];
        if ( (_DWORD)v5 )
        {
          v7 = v5 - 1;
          v8 = v7 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
          v9 = (__int64 *)(v4 + 16LL * v8);
          v10 = *v9;
          if ( v6 == *v9 )
          {
LABEL_8:
            *v9 = -8192;
            --*(_DWORD *)(result + 3184);
            ++*(_DWORD *)(result + 3188);
            v6 = a1[2];
          }
          else
          {
            v11 = 1;
            while ( v10 != -4096 )
            {
              v12 = v11 + 1;
              v8 = v7 & (v11 + v8);
              v9 = (__int64 *)(v4 + 16LL * v8);
              v10 = *v9;
              if ( v6 == *v9 )
                goto LABEL_8;
              v11 = v12;
            }
          }
        }
        *(_BYTE *)(v6 + 1) &= ~1u;
      }
    }
  }
  return result;
}
