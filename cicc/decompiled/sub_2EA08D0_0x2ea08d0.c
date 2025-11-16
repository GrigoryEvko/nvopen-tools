// Function: sub_2EA08D0
// Address: 0x2ea08d0
//
void __fastcall sub_2EA08D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  void *v4; // rdi
  size_t v5; // rdx
  __int64 v6; // rbx
  __int64 i; // r12
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v11; // [rsp+8h] [rbp-D8h] BYREF
  _BYTE *v12; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v13; // [rsp+18h] [rbp-C8h]
  _BYTE v14[192]; // [rsp+20h] [rbp-C0h] BYREF

  v3 = *(unsigned int *)(a1 + 552);
  v4 = *(void **)(a1 + 544);
  v5 = 4 * v3;
  if ( v5 )
    memset(v4, 0, v5);
  if ( *(_DWORD *)(a2 + 72) == 1 )
  {
    v8 = *(_QWORD *)a1;
    v10 = 0;
    v12 = v14;
    v11 = 0;
    v13 = 0x400000000LL;
    v9 = *(__int64 (**)())(*(_QWORD *)v8 + 344LL);
    if ( v9 != sub_2DB1AE0 )
    {
      if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v9)(
              v8,
              a2,
              &v10,
              &v11,
              &v12,
              0)
        && !(_DWORD)v13 )
      {
        sub_2EA08D0(a1, **(_QWORD **)(a2 + 64));
      }
      if ( v12 != v14 )
        _libc_free((unsigned __int64)v12);
    }
  }
  v6 = *(_QWORD *)(a2 + 56);
  for ( i = a2 + 48; i != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    while ( 1 )
    {
      sub_2E9EF00(a1, v6, 1);
      if ( !v6 )
        BUG();
      if ( (*(_BYTE *)v6 & 4) == 0 )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( i == v6 )
        return;
    }
    while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
      v6 = *(_QWORD *)(v6 + 8);
  }
}
