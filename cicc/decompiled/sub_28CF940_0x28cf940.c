// Function: sub_28CF940
// Address: 0x28cf940
//
__int64 __fastcall sub_28CF940(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r14
  unsigned __int64 v9; // r12
  __int64 v10; // rbx
  int v11; // ebx
  __int64 v13; // [rsp+8h] [rbp-48h]
  unsigned __int64 v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x60u, v14, a6);
  v7 = *(_QWORD *)a1;
  v13 = v6;
  v8 = v6;
  v9 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    do
    {
      if ( v8 )
        sub_C8CF70(v8, (void *)(v8 + 32), 8, v7 + 32, v7);
      v7 += 96;
      v8 += 96;
    }
    while ( v9 != v7 );
    v10 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        while ( 1 )
        {
          v9 -= 96LL;
          if ( !*(_BYTE *)(v9 + 28) )
            break;
          if ( v9 == v10 )
            goto LABEL_10;
        }
        _libc_free(*(_QWORD *)(v9 + 8));
      }
      while ( v9 != v10 );
LABEL_10:
      v9 = *(_QWORD *)a1;
    }
  }
  v11 = v14[0];
  if ( a1 + 16 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v11;
  *(_QWORD *)a1 = v13;
  return v13;
}
