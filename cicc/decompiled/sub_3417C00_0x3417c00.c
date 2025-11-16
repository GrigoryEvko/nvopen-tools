// Function: sub_3417C00
// Address: 0x3417c00
//
__int64 *__fastcall sub_3417C00(_QWORD *a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  char v9; // al
  unsigned __int64 v10; // rcx
  __int64 v12; // rsi
  unsigned __int8 *v13; // rax
  unsigned int v14; // edx
  __int64 v15; // r10
  unsigned int v16; // r11d
  __int128 v17; // [rsp-20h] [rbp-80h]
  __int128 v18; // [rsp-10h] [rbp-70h]
  unsigned int v19; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v20; // [rsp+10h] [rbp-50h]
  __int64 *v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h] BYREF
  int v24; // [rsp+28h] [rbp-38h]

  if ( a2 == a4 && (_DWORD)a3 == (_DWORD)a5 )
    return (__int64 *)a4;
  v9 = sub_33CF8A0(a2, a3);
  v10 = a2;
  if ( !v9 )
    return (__int64 *)a4;
  v12 = *(_QWORD *)(a2 + 80);
  v23 = v12;
  if ( v12 )
  {
    sub_B96E90((__int64)&v23, v12, 1);
    v10 = a2;
  }
  *((_QWORD *)&v18 + 1) = a5;
  *(_QWORD *)&v18 = a4;
  *((_QWORD *)&v17 + 1) = a3;
  *(_QWORD *)&v17 = a2;
  v24 = *(_DWORD *)(v10 + 72);
  v13 = sub_3406EB0(a1, 2u, (__int64)&v23, 1, 0, (__int64)&v23, v17, v18);
  v15 = (__int64)v13;
  v16 = v14;
  if ( v23 )
  {
    v19 = v14;
    v20 = v13;
    sub_B91220((__int64)&v23, v23);
    v16 = v19;
    v15 = (__int64)v20;
  }
  v22 = (__int64 *)v15;
  sub_34161C0((__int64)a1, a2, a3, v15, v16);
  sub_33EC010(a1, v22, a2, a3, a4, a5);
  return v22;
}
