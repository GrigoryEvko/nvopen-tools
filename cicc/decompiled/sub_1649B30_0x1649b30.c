// Function: sub_1649B30
// Address: 0x1649b30
//
__int64 __fastcall sub_1649B30(_QWORD *a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // r12
  unsigned __int64 v4; // rcx
  __int64 v5; // rsi
  __int64 v6; // rdx
  int v7; // esi
  unsigned int v8; // r8d
  __int64 *v9; // rdi
  __int64 v10; // r9
  int v11; // edi
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
    result = *(_QWORD *)sub_16498A0(a1[2]);
    v4 = *(_QWORD *)(result + 2648);
    if ( v2 >= v4 )
    {
      v5 = *(unsigned int *)(result + 2664);
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
            *v9 = -16;
            --*(_DWORD *)(result + 2656);
            ++*(_DWORD *)(result + 2660);
            v6 = a1[2];
          }
          else
          {
            v11 = 1;
            while ( v10 != -8 )
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
        *(_BYTE *)(v6 + 17) &= ~1u;
      }
    }
  }
  return result;
}
