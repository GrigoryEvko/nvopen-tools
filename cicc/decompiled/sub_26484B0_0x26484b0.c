// Function: sub_26484B0
// Address: 0x26484b0
//
__int64 __fastcall sub_26484B0(__int64 a1, __int64 a2)
{
  int *v2; // rax
  __int64 v3; // r11
  __int64 v4; // rbx
  unsigned int v5; // r9d
  __int64 v6; // r10
  int v7; // r12d
  int v8; // ecx
  unsigned int v9; // esi
  int *v10; // rdi
  int v11; // r8d
  int v13; // edi
  int v14; // r14d
  __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  int *v16; // [rsp+10h] [rbp-30h]
  int *v17; // [rsp+18h] [rbp-28h]

  sub_22B0690(&v15, (__int64 *)a2);
  v2 = v16;
  v3 = *(_QWORD *)(a2 + 8) + 4LL * *(unsigned int *)(a2 + 24);
  if ( (int *)v3 != v16 )
  {
    v4 = *(_QWORD *)(a1 + 136);
    v5 = 0;
    v6 = *(unsigned int *)(a1 + 152);
    v7 = v6 - 1;
    while ( 1 )
    {
      v8 = *v2;
      if ( (_DWORD)v6 )
      {
        v9 = v7 & (37 * v8);
        v10 = (int *)(v4 + 8LL * v9);
        v11 = *v10;
        if ( v8 == *v10 )
          goto LABEL_5;
        v13 = 1;
        while ( v11 != -1 )
        {
          v14 = v13 + 1;
          v9 = v7 & (v13 + v9);
          v10 = (int *)(v4 + 8LL * v9);
          v11 = *v10;
          if ( v8 == *v10 )
            goto LABEL_5;
          v13 = v14;
        }
      }
      v10 = (int *)(v4 + 8 * v6);
LABEL_5:
      LOBYTE(v5) = *((_BYTE *)v10 + 4) | v5;
      if ( (_BYTE)v5 != 3 )
      {
        for ( ++v2; v17 != v2; ++v2 )
        {
          if ( (unsigned int)*v2 <= 0xFFFFFFFD )
            break;
        }
        if ( (int *)v3 != v2 )
          continue;
      }
      return v5;
    }
  }
  return 0;
}
