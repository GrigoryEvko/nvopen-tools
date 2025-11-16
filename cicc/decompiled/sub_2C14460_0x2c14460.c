// Function: sub_2C14460
// Address: 0x2c14460
//
unsigned __int64 __fastcall sub_2C14460(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // edx
  __int64 v6; // r13
  unsigned __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  __int128 v19; // [rsp-18h] [rbp-118h]
  unsigned __int64 v20; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v21; // [rsp+10h] [rbp-F0h]
  _QWORD v22[2]; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v23[3]; // [rsp+30h] [rbp-D0h] BYREF
  char *v24; // [rsp+48h] [rbp-B8h]
  char v25; // [rsp+58h] [rbp-A8h] BYREF
  char *v26; // [rsp+78h] [rbp-88h]
  char v27; // [rsp+88h] [rbp-78h] BYREF

  v5 = *(unsigned __int8 *)(a1 + 160);
  if ( (unsigned int)(v5 - 13) <= 0x11 )
  {
    if ( *(_QWORD *)(a1 + 136) )
    {
      v6 = sub_2BFD6A0(a3 + 16, a1 + 96);
      if ( !(unsigned __int8)sub_2C46C30(a1 + 96) )
        v6 = sub_2AAEDF0(v6, a2);
      return sub_DFD800(*(_QWORD *)a3, *(unsigned __int8 *)(a1 + 160), v6, *(_DWORD *)(a3 + 176), 0, 0, 0, 0, 0, 0);
    }
    return 0;
  }
  if ( (_BYTE)v5 == 85 )
  {
    v17 = sub_2BFD6A0(a3 + 16, a1 + 96);
    v18 = sub_2AAEDF0(v17, a2);
    BYTE4(v23[0]) = 0;
    return sub_DFDC10(*(__int64 **)a3, 29, v18, v23[0]);
  }
  if ( (_BYTE)v5 != 86 )
    return 0;
  v8 = sub_2BFD6A0(a3 + 16, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL));
  v9 = sub_2AAEDF0(v8, a2);
  v10 = *(_QWORD **)(a3 + 64);
  v22[0] = v9;
  v11 = sub_BCB2A0(v10);
  v12 = *(_QWORD **)(a3 + 64);
  v22[1] = v11;
  v13 = sub_BCB2E0(v12);
  *((_QWORD *)&v19 + 1) = 1;
  *(_QWORD *)&v19 = 0;
  sub_DF8CB0((__int64)v23, 145, v13, (char *)v22, 2, 0, 0, v19);
  v14 = sub_DFD690(*(_QWORD *)a3, (__int64)v23);
  v15 = sub_2BFD6A0(a3 + 16, **(_QWORD **)(a1 + 48));
  sub_2AAEDF0(v15, a2);
  v16 = sub_DFD330(*(__int64 **)a3);
  result = v16 + v14;
  if ( __OFADD__(v16, v14) )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v16 <= 0 )
      result = 0x8000000000000000LL;
  }
  if ( v26 != &v27 )
  {
    v20 = result;
    _libc_free((unsigned __int64)v26);
    result = v20;
  }
  if ( v24 != &v25 )
  {
    v21 = result;
    _libc_free((unsigned __int64)v24);
    return v21;
  }
  return result;
}
