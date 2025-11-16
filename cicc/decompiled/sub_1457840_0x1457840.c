// Function: sub_1457840
// Address: 0x1457840
//
__int64 __fastcall sub_1457840(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rcx
  int v5; // edx
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  int v11; // edx
  int v12; // r9d

  result = sub_157F0B0(a2);
  if ( !result )
  {
    v4 = *(_QWORD *)(a1 + 64);
    v5 = *(_DWORD *)(v4 + 24);
    if ( v5 )
    {
      v6 = *(_QWORD *)(v4 + 8);
      v7 = v5 - 1;
      v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
      {
LABEL_4:
        if ( v9[1] )
          return sub_13FC470(v9[1]);
        else
          return 0;
      }
      else
      {
        v11 = 1;
        while ( v10 != -8 )
        {
          v12 = v11 + 1;
          v8 = v7 & (v11 + v8);
          v9 = (__int64 *)(v6 + 16LL * v8);
          v10 = *v9;
          if ( a2 == *v9 )
            goto LABEL_4;
          v11 = v12;
        }
      }
    }
  }
  return result;
}
