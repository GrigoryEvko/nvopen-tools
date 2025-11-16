// Function: sub_31C3960
// Address: 0x31c3960
//
__int64 __fastcall sub_31C3960(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rdi
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // r14
  __int64 v14; // rax
  unsigned __int64 v15; // r15
  __int64 v16; // rdi
  int v17; // r13d
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  unsigned __int64 v21; // [rsp+18h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v20 = a1 + 16;
  v19 = sub_C8D7D0(a1, a1 + 16, a2, 0x58u, v22, a6);
  v7 = v19;
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    do
    {
      while ( 1 )
      {
        if ( v7 )
        {
          *(_DWORD *)v7 = *(_DWORD *)v8;
          *(_QWORD *)(v7 + 8) = *(_QWORD *)(v8 + 8);
          v10 = *(_QWORD *)(v8 + 16);
          *(_DWORD *)(v7 + 32) = 0;
          *(_QWORD *)(v7 + 16) = v10;
          *(_QWORD *)(v7 + 24) = v7 + 40;
          *(_DWORD *)(v7 + 36) = 6;
          if ( *(_DWORD *)(v8 + 32) )
            break;
        }
        v8 += 88LL;
        v7 += 88;
        if ( v9 == v8 )
          goto LABEL_7;
      }
      v11 = v7 + 24;
      v21 = v8;
      v7 += 88;
      sub_31C3510(v11, v8 + 24);
      v8 = v21 + 88;
    }
    while ( v9 != v21 + 88 );
LABEL_7:
    v12 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v13 = *(_QWORD *)(v9 - 64);
        v14 = *(unsigned int *)(v9 - 56);
        v9 -= 88LL;
        v15 = v13 + 8 * v14;
        if ( v13 != v15 )
        {
          do
          {
            v16 = *(_QWORD *)(v15 - 8);
            v15 -= 8LL;
            if ( v16 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
          }
          while ( v13 != v15 );
          v13 = *(_QWORD *)(v9 + 24);
        }
        if ( v13 != v9 + 40 )
          _libc_free(v13);
      }
      while ( v9 != v12 );
      v9 = *(_QWORD *)a1;
    }
  }
  v17 = v22[0];
  if ( v20 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v17;
  *(_QWORD *)a1 = v19;
  return v19;
}
