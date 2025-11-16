// Function: sub_190C060
// Address: 0x190c060
//
void __fastcall sub_190C060(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 *v7; // rax
  __int64 *v8; // rbx
  unsigned __int64 v9; // rcx
  char *v10; // rbx
  char *v11; // r12
  char *v12; // rdi
  __int64 v14; // [rsp+28h] [rbp-278h]
  _QWORD v15[2]; // [rsp+30h] [rbp-270h] BYREF
  _QWORD v16[2]; // [rsp+40h] [rbp-260h] BYREF
  _QWORD *v17; // [rsp+50h] [rbp-250h]
  _QWORD v18[6]; // [rsp+60h] [rbp-240h] BYREF
  _QWORD v19[11]; // [rsp+90h] [rbp-210h] BYREF
  char *v20; // [rsp+E8h] [rbp-1B8h]
  unsigned int v21; // [rsp+F0h] [rbp-1B0h]
  char v22; // [rsp+F8h] [rbp-1A8h] BYREF

  sub_15CA5C0((__int64)v19, (__int64)"gvn", (__int64)"LoadClobbered", 13, (__int64)a1);
  sub_15CAB20((__int64)v19, "load of type ", 0xDu);
  sub_15C9730((__int64)v15, "Type", 4, *a1);
  v5 = sub_17C21B0((__int64)v19, (__int64)v15);
  sub_15CAB20(v5, " not eliminated", 0xFu);
  sub_15CA8D0(v5);
  if ( v17 != v18 )
    j_j___libc_free_0(v17, v18[0] + 1LL);
  if ( (_QWORD *)v15[0] != v16 )
    j_j___libc_free_0(v15[0], v16[0] + 1LL);
  v14 = 0;
  if ( *(_QWORD *)(*(a1 - 3) + 8) )
  {
    v6 = *(_QWORD *)(*(a1 - 3) + 8);
    do
    {
      v7 = sub_1648700(v6);
      v8 = v7;
      if ( a1 != v7 && (unsigned __int8)(*((_BYTE *)v7 + 16) - 54) <= 1u && sub_15CCEE0(a3, (__int64)v7, (__int64)a1) )
      {
        if ( v14 )
          v8 = 0;
        v14 = (__int64)v8;
      }
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v6 );
    if ( v14 )
    {
      sub_15CAB20((__int64)v19, " in favor of ", 0xDu);
      sub_15C9340((__int64)v15, "OtherAccess", 0xBu, v14);
      sub_17C21B0((__int64)v19, (__int64)v15);
      if ( v17 != v18 )
        j_j___libc_free_0(v17, v18[0] + 1LL);
      if ( (_QWORD *)v15[0] != v16 )
        j_j___libc_free_0(v15[0], v16[0] + 1LL);
    }
  }
  sub_15CAB20((__int64)v19, " because it is clobbered by ", 0x1Cu);
  v9 = 0;
  if ( ((unsigned __int8)a2 & 7u) < 3 )
    v9 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  sub_15C9340((__int64)v15, "ClobberedBy", 0xBu, v9);
  sub_17C21B0((__int64)v19, (__int64)v15);
  if ( v17 != v18 )
    j_j___libc_free_0(v17, v18[0] + 1LL);
  if ( (_QWORD *)v15[0] != v16 )
    j_j___libc_free_0(v15[0], v16[0] + 1LL);
  sub_143AA50(a4, (__int64)v19);
  v10 = v20;
  v19[0] = &unk_49ECF68;
  v11 = &v20[88 * v21];
  if ( v20 != v11 )
  {
    do
    {
      v11 -= 88;
      v12 = (char *)*((_QWORD *)v11 + 4);
      if ( v12 != v11 + 48 )
        j_j___libc_free_0(v12, *((_QWORD *)v11 + 6) + 1LL);
      if ( *(char **)v11 != v11 + 16 )
        j_j___libc_free_0(*(_QWORD *)v11, *((_QWORD *)v11 + 2) + 1LL);
    }
    while ( v10 != v11 );
    v11 = v20;
  }
  if ( v11 != &v22 )
    _libc_free((unsigned __int64)v11);
}
