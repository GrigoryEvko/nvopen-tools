// Function: sub_1E20630
// Address: 0x1e20630
//
void __fastcall sub_1E20630(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _BYTE **a5, int a6)
{
  __int64 v7; // rdx
  void *v8; // rdi
  size_t v9; // rdx
  __int64 v10; // rbx
  __int64 i; // r12
  __int64 v12; // rdi
  __int64 (*v13)(); // rax
  __int64 v14; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v15; // [rsp+8h] [rbp-D8h] BYREF
  _BYTE *v16; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v17; // [rsp+18h] [rbp-C8h]
  _BYTE v18[192]; // [rsp+20h] [rbp-C0h] BYREF

  v7 = *(unsigned int *)(a1 + 944);
  v8 = *(void **)(a1 + 936);
  v9 = 4 * v7;
  if ( v9 )
    memset(v8, 0, v9);
  if ( (unsigned int)((__int64)(*(_QWORD *)(a2 + 72) - *(_QWORD *)(a2 + 64)) >> 3) == 1 )
  {
    v12 = *(_QWORD *)(a1 + 232);
    v14 = 0;
    v16 = v18;
    a5 = &v16;
    v15 = 0;
    v17 = 0x400000000LL;
    v13 = *(__int64 (**)())(*(_QWORD *)v12 + 264LL);
    if ( v13 != sub_1D820E0 )
    {
      if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v13)(
              v12,
              a2,
              &v14,
              &v15,
              &v16,
              0)
        && !(_DWORD)v17 )
      {
        sub_1E20630(a1, **(_QWORD **)(a2 + 64));
      }
      if ( v16 != v18 )
        _libc_free((unsigned __int64)v16);
    }
  }
  v10 = *(_QWORD *)(a2 + 32);
  for ( i = a2 + 24; i != v10; v10 = *(_QWORD *)(v10 + 8) )
  {
    while ( 1 )
    {
      sub_1E20560(a1, v10, 1u, a4, (__int64)a5, a6);
      if ( !v10 )
        BUG();
      if ( (*(_BYTE *)v10 & 4) == 0 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( i == v10 )
        return;
    }
    while ( (*(_BYTE *)(v10 + 46) & 8) != 0 )
      v10 = *(_QWORD *)(v10 + 8);
  }
}
