// Function: sub_B1DF70
// Address: 0xb1df70
//
__int64 __fastcall sub_B1DF70(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  char **v4; // rsi
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rdi
  int v14; // ebx
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v4 = (char **)(a1 + 16);
  v6 = sub_C8D7D0(a1, a1 + 16, a2, 56, v17);
  v7 = *(_QWORD *)a1;
  v16 = v6;
  v8 = v6;
  v9 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    do
    {
      while ( 1 )
      {
        if ( v8 )
        {
          *(_DWORD *)v8 = *(_DWORD *)v7;
          *(_DWORD *)(v8 + 4) = *(_DWORD *)(v7 + 4);
          *(_DWORD *)(v8 + 8) = *(_DWORD *)(v7 + 8);
          *(_DWORD *)(v8 + 12) = *(_DWORD *)(v7 + 12);
          v10 = *(_QWORD *)(v7 + 16);
          *(_DWORD *)(v8 + 32) = 0;
          *(_QWORD *)(v8 + 16) = v10;
          *(_QWORD *)(v8 + 24) = v8 + 40;
          *(_DWORD *)(v8 + 36) = 4;
          if ( *(_DWORD *)(v7 + 32) )
            break;
        }
        v7 += 56;
        v8 += 56;
        if ( v9 == v7 )
          goto LABEL_7;
      }
      v4 = (char **)(v7 + 24);
      v11 = v8 + 24;
      v7 += 56;
      v8 += 56;
      sub_B189E0(v11, v4);
    }
    while ( v9 != v7 );
LABEL_7:
    v12 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v9 -= 56;
        v13 = *(_QWORD *)(v9 + 24);
        if ( v13 != v9 + 40 )
          _libc_free(v13, v4);
      }
      while ( v9 != v12 );
      v9 = *(_QWORD *)a1;
    }
  }
  v14 = v17[0];
  if ( v3 != v9 )
    _libc_free(v9, v4);
  *(_DWORD *)(a1 + 12) = v14;
  *(_QWORD *)a1 = v16;
  return v16;
}
