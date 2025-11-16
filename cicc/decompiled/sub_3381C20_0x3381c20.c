// Function: sub_3381C20
// Address: 0x3381c20
//
void __fastcall sub_3381C20(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // r15
  _QWORD *v6; // r12
  _DWORD *v7; // rsi
  int v8; // eax
  char v9; // bl
  unsigned __int8 *v10; // r13
  char v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // eax
  int v17; // [rsp+24h] [rbp-11Ch]
  unsigned __int8 *v18; // [rsp+38h] [rbp-108h] BYREF
  _DWORD *v19; // [rsp+40h] [rbp-100h] BYREF
  _DWORD *v20; // [rsp+48h] [rbp-F8h] BYREF
  _BYTE *v21; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v22; // [rsp+58h] [rbp-E8h]
  _BYTE v23[32]; // [rsp+60h] [rbp-E0h] BYREF
  _DWORD *v24; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+88h] [rbp-B8h]
  _BYTE v26[176]; // [rsp+90h] [rbp-B0h] BYREF

  v3 = a1;
  v5 = *(_QWORD *)(a3 + 8);
  v6 = *(_QWORD **)(a3 + 16);
  v18 = a2;
  v7 = *(_DWORD **)(a3 + 24);
  v19 = v7;
  if ( v7 )
  {
    sub_B96E90((__int64)&v19, (__int64)v7, 1);
    v8 = *(_DWORD *)a3;
    v24 = v19;
    v17 = v8;
    if ( v19 )
      sub_B96E90((__int64)&v24, (__int64)v19, 1);
  }
  else
  {
    v15 = *(_DWORD *)a3;
    v24 = 0;
    v17 = v15;
  }
  v9 = sub_3380DB0(a1, (__int64 *)&v18, 1, v5, (__int64)v6, &v24, v17, 0);
  if ( v24 )
    sub_B91220((__int64)&v24, (__int64)v24);
  if ( v9 )
  {
LABEL_32:
    if ( v19 )
      sub_B91220((__int64)&v19, (__int64)v19);
  }
  else
  {
    if ( *v18 > 0x1Cu )
    {
      v10 = v18;
      while ( 1 )
      {
        v24 = v26;
        v25 = 0x1000000000LL;
        v21 = v23;
        v22 = 0x400000000LL;
        v12 = sub_AF4EB0((__int64)v6);
        v18 = (unsigned __int8 *)sub_F53E50(v10, v12, (__int64)&v24, (__int64)&v21);
        if ( !v18 || (_DWORD)v22 )
          break;
        v6 = (_QWORD *)sub_B0DBA0(v6, v24, (unsigned int)v25, 0, 1);
        v20 = v19;
        if ( v19 )
          sub_B96E90((__int64)&v20, (__int64)v19, 1);
        v11 = sub_3380DB0(a1, (__int64 *)&v18, 1, v5, (__int64)v6, &v20, v17, 0);
        if ( v20 )
          sub_B91220((__int64)&v20, (__int64)v20);
        if ( v11 )
        {
          if ( v21 != v23 )
            _libc_free((unsigned __int64)v21);
          if ( v24 != (_DWORD *)v26 )
            _libc_free((unsigned __int64)v24);
          goto LABEL_32;
        }
        if ( v21 != v23 )
          _libc_free((unsigned __int64)v21);
        if ( v24 != (_DWORD *)v26 )
          _libc_free((unsigned __int64)v24);
        v10 = v18;
        if ( *v18 <= 0x1Cu )
        {
          v3 = a1;
          goto LABEL_25;
        }
      }
      v3 = a1;
      if ( v21 != v23 )
        _libc_free((unsigned __int64)v21);
      if ( v24 != (_DWORD *)v26 )
        _libc_free((unsigned __int64)v24);
    }
LABEL_25:
    v13 = sub_ACADE0(*((__int64 ***)a2 + 1));
    v14 = sub_33E5DB0(*(_QWORD *)(v3 + 864), v5, v6, v13, &v19, *(unsigned int *)(v3 + 848));
    sub_33F99B0(*(_QWORD *)(v3 + 864), v14, 0);
    if ( v19 )
      sub_B91220((__int64)&v19, (__int64)v19);
  }
}
