// Function: sub_C21B60
// Address: 0xc21b60
//
__int64 __fastcall sub_C21B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // rdi
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rdx
  int v15; // ecx
  int v16; // r9d

  v5 = sub_EF7600(*(_QWORD *)(a2 + 8), a3, a4);
  if ( v5 )
  {
    v6 = *(_DWORD *)(a2 + 40);
    v7 = *(_QWORD *)(a2 + 24);
    if ( v6 )
    {
      v8 = v6 - 1;
      v9 = (v6 - 1) & (((0xBF58476D1CE4E5B9LL * v5) >> 31) ^ (484763065 * v5));
      v10 = (__int64 *)(v7 + 24LL * v9);
      v11 = *v10;
      if ( v5 == *v10 )
      {
LABEL_4:
        v12 = v10[2];
        if ( v12 )
        {
          v13 = v10[1];
          *(_QWORD *)(a1 + 8) = v12;
          *(_BYTE *)(a1 + 16) = 1;
          *(_QWORD *)a1 = v13;
          return a1;
        }
      }
      else
      {
        v15 = 1;
        while ( v11 != -1 )
        {
          v16 = v15 + 1;
          v9 = v8 & (v15 + v9);
          v10 = (__int64 *)(v7 + 24LL * v9);
          v11 = *v10;
          if ( v5 == *v10 )
            goto LABEL_4;
          v15 = v16;
        }
      }
    }
  }
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
