// Function: sub_F55740
// Address: 0xf55740
//
__int64 __fastcall sub_F55740(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v9; // rdi
  unsigned __int64 *v10; // rdx
  __int64 v11; // rdi
  unsigned __int64 *v12; // rdx
  __int64 v13; // rax
  unsigned __int64 *v14; // rdx
  unsigned __int64 *v15; // rdx
  bool v16; // [rsp+Fh] [rbp-71h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  unsigned __int64 v19; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 v20; // [rsp+28h] [rbp-58h] BYREF
  unsigned __int64 *v21; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 *v22; // [rsp+38h] [rbp-48h]
  unsigned __int64 *v23; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 *v24; // [rsp+48h] [rbp-38h]

  if ( (*(_BYTE *)(a1 + 7) & 8) == 0 )
    return 0;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_QWORD *)(a2 + 8);
  v9 = sub_B43CA0(a1);
  if ( v6 == v5 )
    return sub_F54F50(a1, (unsigned __int8 *)a2, a3, a4, sub_F4EE20, (__int64)&v21, sub_F4EFC0, (__int64)&v23);
  if ( (*(_BYTE *)(v5 + 8) & 0xFD) != 0xC || (*(_BYTE *)(v6 + 8) & 0xFD) != 0xC )
    return 0;
  v17 = v9 + 312;
  v23 = (unsigned __int64 *)sub_9208B0(v9 + 312, v6);
  v24 = v10;
  v16 = 0;
  v21 = (unsigned __int64 *)sub_9208B0(v9 + 312, v5);
  v11 = v9 + 312;
  v22 = v12;
  if ( v21 == v23 )
    v16 = (_BYTE)v22 == (unsigned __int8)v24;
  if ( *(_BYTE *)(v5 + 8) == 14 )
  {
    if ( *((_BYTE *)sub_AE2980(v11, *(_DWORD *)(v5 + 8) >> 8) + 16) )
      goto LABEL_10;
    v11 = v17;
  }
  if ( (*(_BYTE *)(v6 + 8) != 14 || !*((_BYTE *)sub_AE2980(v11, *(_DWORD *)(v6 + 8) >> 8) + 16)) && v16 )
    return sub_F54F50(a1, (unsigned __int8 *)a2, a3, a4, sub_F4EE20, (__int64)&v21, sub_F4EFC0, (__int64)&v23);
LABEL_10:
  if ( *(_BYTE *)(v5 + 8) != 12 || *(_BYTE *)(v6 + 8) != 12 )
    return 0;
  v13 = sub_BCAE30(v5);
  v24 = v14;
  v23 = (unsigned __int64 *)v13;
  v19 = sub_CA1930(&v23);
  v23 = (unsigned __int64 *)sub_BCAE30(v6);
  v24 = v15;
  v20 = sub_CA1930(&v23);
  if ( v20 > v19 )
    return sub_F54F50(a1, (unsigned __int8 *)a2, a3, a4, sub_F4EE20, (__int64)&v21, sub_F4EFC0, (__int64)&v23);
  v22 = &v19;
  v24 = &v19;
  v21 = &v20;
  v23 = &v20;
  return sub_F54F50(
           a1,
           (unsigned __int8 *)a2,
           a3,
           a4,
           (__int64 (__fastcall *)(__int64, __int64))sub_F4F7F0,
           (__int64)&v21,
           (__int64 (__fastcall *)(__int64, __int64))sub_F4F720,
           (__int64)&v23);
}
