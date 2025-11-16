// Function: sub_2445EC0
// Address: 0x2445ec0
//
__int64 __fastcall sub_2445EC0(
        __int64 a1,
        unsigned __int8 *a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        char a5,
        __int64 *a6)
{
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int8 *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r9
  _QWORD *v19; // r8
  __int64 v20; // r12
  unsigned __int64 v22; // [rsp+8h] [rbp-68h] BYREF
  unsigned __int64 v23; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int8 *v24; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v25[10]; // [rsp+20h] [rbp-50h] BYREF

  v8 = a4 - a3;
  v9 = a3;
  v23 = a3;
  v24 = a2;
  v22 = a4;
  v10 = sub_BD5C60(a1);
  v11 = v8;
  if ( v9 >= v8 )
    v11 = v9;
  v25[0] = v10;
  if ( v11 > 0xFFFFFFFE )
  {
    v12 = v11 / 0xFFFFFFFF + 1;
    v8 /= v12;
    v9 /= v12;
  }
  v13 = sub_B8C2F0(v25, v9, v8, 0);
  v14 = v24;
  v15 = sub_29A5A50(a1, v24, v13);
  v19 = v25;
  v20 = v15;
  if ( a5 )
  {
    v14 = (unsigned __int8 *)v25;
    LODWORD(v25[0]) = v23;
    sub_BC8EC0(v15, (unsigned int *)v25, 1, 0);
  }
  if ( a6 )
  {
    v25[0] = a1;
    v25[1] = &v24;
    v25[2] = &v23;
    v25[3] = &v22;
    sub_2445A80(a6, (__int64)v14, v16, v17, (__int64)v19, v18, a1, &v24, &v23, &v22);
  }
  return v20;
}
