// Function: sub_354B380
// Address: 0x354b380
//
__int64 __fastcall sub_354B380(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  unsigned __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // rbx
  int v11; // ebx
  __int64 v13; // [rsp+8h] [rbp-48h]
  unsigned __int64 v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x40u, v14, a6);
  v7 = *(_QWORD *)a1;
  v13 = v6;
  v8 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v6;
    do
    {
      if ( v9 )
        sub_C8CF70(v9, (void *)(v9 + 32), 4, v7 + 32, v7);
      v7 += 64;
      v9 += 64;
    }
    while ( v8 != v7 );
    v10 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        while ( 1 )
        {
          v8 -= 64LL;
          if ( !*(_BYTE *)(v8 + 28) )
            break;
          if ( v8 == v10 )
            goto LABEL_11;
        }
        _libc_free(*(_QWORD *)(v8 + 8));
      }
      while ( v8 != v10 );
LABEL_11:
      v8 = *(_QWORD *)a1;
    }
  }
  v11 = v14[0];
  if ( a1 + 16 != v8 )
    _libc_free(v8);
  *(_DWORD *)(a1 + 12) = v11;
  *(_QWORD *)a1 = v13;
  return v13;
}
