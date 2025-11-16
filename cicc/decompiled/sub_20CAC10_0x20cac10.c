// Function: sub_20CAC10
// Address: 0x20cac10
//
__int64 __fastcall sub_20CAC10(__int64 *a1, __int64 a2, __int64 **a3, __int64 a4, int a5, __int64 *a6, __int64 *a7)
{
  __int16 v9; // bx
  int v10; // r9d
  _QWORD *v11; // rax
  _QWORD *v12; // r12
  __int64 v13; // rdi
  unsigned __int64 *v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int8 *v20; // rsi
  __int64 result; // rax
  unsigned int v24; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v25; // [rsp+28h] [rbp-58h] BYREF
  __int64 v26[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v27; // [rsp+40h] [rbp-40h]

  v9 = a5;
  switch ( a5 )
  {
    case 0:
    case 1:
    case 3:
    case 7:
      v10 = a5;
      break;
    case 2:
    case 5:
      v10 = 2;
      break;
    case 4:
    case 6:
      v10 = 4;
      break;
  }
  v24 = v10;
  v27 = 257;
  v11 = sub_1648A60(64, 3u);
  v12 = v11;
  if ( v11 )
    sub_15F99E0((__int64)v11, a2, a3, a4, v9, v24, 1, 0);
  v13 = a1[1];
  if ( v13 )
  {
    v14 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v13 + 40, (__int64)v12);
    v15 = v12[3];
    v16 = *v14;
    v12[4] = v14;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    v12[3] = v16 | v15 & 7;
    *(_QWORD *)(v16 + 8) = v12 + 3;
    *v14 = *v14 & 7 | (unsigned __int64)(v12 + 3);
  }
  sub_164B780((__int64)v12, v26);
  v17 = *a1;
  if ( *a1 )
  {
    v25 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v25, v17, 2);
    v18 = v12[6];
    v19 = (__int64)(v12 + 6);
    if ( v18 )
    {
      sub_161E7C0((__int64)(v12 + 6), v18);
      v19 = (__int64)(v12 + 6);
    }
    v20 = v25;
    v12[6] = v25;
    if ( v20 )
      sub_1623210((__int64)&v25, v20, v19);
  }
  v26[0] = (__int64)"success";
  v27 = 259;
  LODWORD(v25) = 1;
  *a6 = sub_12A9E60(a1, (__int64)v12, (__int64)&v25, 1, (__int64)v26);
  v26[0] = (__int64)"newloaded";
  v27 = 259;
  LODWORD(v25) = 0;
  result = sub_12A9E60(a1, (__int64)v12, (__int64)&v25, 1, (__int64)v26);
  *a7 = result;
  return result;
}
