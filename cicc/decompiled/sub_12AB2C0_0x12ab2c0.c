// Function: sub_12AB2C0
// Address: 0x12ab2c0
//
__int64 __fastcall sub_12AB2C0(__int64 a1, _QWORD *a2, __int64 a3, int a4, int a5, int a6)
{
  char *v10; // r15
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  _QWORD *v14; // r15
  __int64 v15; // rdi
  unsigned __int64 *v16; // r14
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rsi
  _QWORD *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // r15
  __int64 v23; // rax
  _QWORD *v24; // r12
  __int64 v25; // rdi
  unsigned __int64 *v26; // r15
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rsi
  char *v33; // [rsp+10h] [rbp-70h]
  char *v34; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+28h] [rbp-58h] BYREF
  _BYTE v37[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v38; // [rsp+40h] [rbp-40h]

  v10 = sub_128F980((__int64)a2, a3);
  v11 = sub_12AB0E0(a2, a4);
  sub_12A8F50(a2 + 6, v11, (__int64)v10, 0);
  v33 = sub_128F980((__int64)a2, *(_QWORD *)(a3 + 16));
  v38 = 257;
  v12 = sub_12AB0E0(a2, a5);
  v13 = sub_1648A60(64, 2);
  v14 = (_QWORD *)v13;
  if ( v13 )
    sub_15F9650(v13, v12, v33, 0, 0);
  v15 = a2[7];
  if ( v15 )
  {
    v16 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v15 + 40, v14);
    v17 = v14[3];
    v18 = *v16;
    v14[4] = v16;
    v18 &= 0xFFFFFFFFFFFFFFF8LL;
    v14[3] = v18 | v17 & 7;
    *(_QWORD *)(v18 + 8) = v14 + 3;
    *v16 = *v16 & 7 | (unsigned __int64)(v14 + 3);
  }
  sub_164B780(v14, v37);
  v19 = a2[6];
  if ( v19 )
  {
    v36 = a2[6];
    sub_1623A60(&v36, v19, 2);
    v20 = v14 + 6;
    if ( v14[6] )
    {
      sub_161E7C0(v14 + 6);
      v20 = v14 + 6;
    }
    v21 = v36;
    v14[6] = v36;
    if ( v21 )
      sub_1623210(&v36, v21, v20);
  }
  v34 = sub_128F980((__int64)a2, *(_QWORD *)(*(_QWORD *)(a3 + 16) + 16LL));
  v22 = sub_12AB0E0(a2, a6);
  v38 = 257;
  v23 = sub_1648A60(64, 2);
  v24 = (_QWORD *)v23;
  if ( v23 )
    sub_15F9650(v23, v22, v34, 0, 0);
  v25 = a2[7];
  if ( v25 )
  {
    v26 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v25 + 40, v24);
    v27 = v24[3];
    v28 = *v26;
    v24[4] = v26;
    v28 &= 0xFFFFFFFFFFFFFFF8LL;
    v24[3] = v28 | v27 & 7;
    *(_QWORD *)(v28 + 8) = v24 + 3;
    *v26 = *v26 & 7 | (unsigned __int64)(v24 + 3);
  }
  sub_164B780(v24, v37);
  v29 = a2[6];
  if ( v29 )
  {
    v36 = a2[6];
    sub_1623A60(&v36, v29, 2);
    if ( v24[6] )
      sub_161E7C0(v24 + 6);
    v30 = v36;
    v24[6] = v36;
    if ( v30 )
      sub_1623210(&v36, v30, v24 + 6);
  }
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
