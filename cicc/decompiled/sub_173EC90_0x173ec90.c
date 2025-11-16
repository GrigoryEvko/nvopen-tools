// Function: sub_173EC90
// Address: 0x173ec90
//
__int64 __fastcall sub_173EC90(
        __int64 a1,
        _QWORD *a2,
        __int64 *a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  int v12; // r8d
  __int64 v14; // rax
  __int64 v15; // r11
  __int64 *v16; // rcx
  __int64 *v17; // rdx
  int v18; // esi
  __int64 v19; // rax
  _QWORD *v20; // rax
  int v21; // r8d
  __int64 v22; // r11
  _QWORD *v23; // r10
  __int64 v24; // r12
  __int64 *v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rax
  int v28; // r8d
  __int64 v29; // r9
  char v30; // al
  int v31; // r13d
  __int64 v32; // rdi
  __int64 *v33; // r13
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  bool v37; // zf
  __int64 v38; // rsi
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  _QWORD *v42; // rax
  _QWORD *v43; // [rsp+8h] [rbp-78h]
  __int64 *v44; // [rsp+10h] [rbp-70h]
  __int64 v45; // [rsp+10h] [rbp-70h]
  __int64 v46; // [rsp+18h] [rbp-68h]
  __int64 v47; // [rsp+18h] [rbp-68h]
  __int64 v48; // [rsp+18h] [rbp-68h]
  int v50; // [rsp+28h] [rbp-58h]
  __int64 v51; // [rsp+28h] [rbp-58h]
  _QWORD v52[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v53; // [rsp+40h] [rbp-40h]

  v12 = a4;
  v14 = *a2;
  v53 = 257;
  v15 = *(_QWORD *)(v14 + 24);
  v16 = &a5[7 * a6];
  if ( a5 == v16 )
  {
    v48 = *(_QWORD *)(v14 + 24);
    v42 = sub_1648AB0(72, (int)a4 + 1, 16 * (int)a6);
    v28 = a4 + 1;
    v22 = v48;
    v29 = a4;
    v23 = a2;
    v24 = (__int64)v42;
    if ( v42 )
    {
      v51 = (__int64)v42;
LABEL_8:
      v45 = (__int64)v23;
      v47 = v22;
      sub_15F1EA0(v24, **(_QWORD **)(v22 + 16), 54, v24 - 24 * v29 - 24, v28, 0);
      *(_QWORD *)(v24 + 56) = 0;
      sub_15F5B40(v24, v47, v45, a3, a4, (__int64)v52, a5, a6);
      goto LABEL_9;
    }
  }
  else
  {
    v17 = a5;
    v18 = 0;
    do
    {
      v19 = v17[5] - v17[4];
      v17 += 7;
      v18 += v19 >> 3;
    }
    while ( v16 != v17 );
    v43 = a2;
    v44 = &a5[7 * a6];
    v46 = v15;
    v50 = v12 + 1;
    v20 = sub_1648AB0(72, v12 + 1 + v18, 16 * (int)a6);
    v21 = v50;
    v22 = v46;
    v23 = v43;
    v24 = (__int64)v20;
    if ( v20 )
    {
      v51 = (__int64)v20;
      v25 = a5;
      LODWORD(v26) = 0;
      do
      {
        v27 = v25[5] - v25[4];
        v25 += 7;
        v26 = (unsigned int)(v27 >> 3) + (unsigned int)v26;
      }
      while ( v44 != v25 );
      v28 = v26 + v21;
      v29 = a4 + v26;
      goto LABEL_8;
    }
  }
  v51 = 0;
  v24 = 0;
LABEL_9:
  v30 = *(_BYTE *)(*(_QWORD *)v24 + 8LL);
  if ( v30 == 16 )
    v30 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v24 + 16LL) + 8LL);
  if ( (unsigned __int8)(v30 - 1) <= 5u || *(_BYTE *)(v24 + 16) == 76 )
  {
    v31 = *(_DWORD *)(a1 + 40);
    if ( a8 || (a8 = *(_QWORD *)(a1 + 32)) != 0 )
      sub_1625C10(v24, 3, a8);
    sub_15F2440(v24, v31);
  }
  v32 = *(_QWORD *)(a1 + 8);
  if ( v32 )
  {
    v33 = *(__int64 **)(a1 + 16);
    sub_157E9D0(v32 + 40, v24);
    v34 = *(_QWORD *)(v24 + 24);
    v35 = *v33;
    *(_QWORD *)(v24 + 32) = v33;
    v35 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v24 + 24) = v35 | v34 & 7;
    *(_QWORD *)(v35 + 8) = v24 + 24;
    *v33 = *v33 & 7 | (v24 + 24);
  }
  sub_164B780(v51, a7);
  v37 = *(_QWORD *)(a1 + 80) == 0;
  v52[0] = v24;
  if ( v37 )
    sub_4263D6(v51, a7, v36);
  (*(void (__fastcall **)(__int64, _QWORD *))(a1 + 88))(a1 + 64, v52);
  v38 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v52[0] = *(_QWORD *)a1;
    sub_1623A60((__int64)v52, v38, 2);
    v39 = *(_QWORD *)(v24 + 48);
    if ( v39 )
      sub_161E7C0(v24 + 48, v39);
    v40 = (unsigned __int8 *)v52[0];
    *(_QWORD *)(v24 + 48) = v52[0];
    if ( v40 )
      sub_1623210((__int64)v52, v40, v24 + 48);
  }
  return v24;
}
