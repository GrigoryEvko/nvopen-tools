// Function: sub_1817AE0
// Address: 0x1817ae0
//
_QWORD *__fastcall sub_1817AE0(_QWORD *a1, unsigned int a2, __int64 a3)
{
  _QWORD *v5; // rax
  unsigned __int8 *v6; // rsi
  __int64 v7; // rax
  _BYTE *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  bool v12; // cc
  __int64 v13; // rax
  unsigned __int8 *v14; // rsi
  _QWORD *v15; // r12
  __int64 v17; // rax
  __int64 v18; // r14
  _QWORD *v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // r11
  __int64 v24; // rax
  unsigned __int64 *v25; // rbx
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rsi
  unsigned __int8 *v29; // rsi
  int v30; // [rsp+4h] [rbp-ECh]
  __int64 v31; // [rsp+8h] [rbp-E8h]
  unsigned __int8 *v32; // [rsp+18h] [rbp-D8h] BYREF
  __int64 *v33; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v34; // [rsp+28h] [rbp-C8h]
  __int64 v35; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v36; // [rsp+40h] [rbp-B0h]
  unsigned __int8 *v37[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v38; // [rsp+60h] [rbp-90h]
  unsigned __int8 *v39; // [rsp+70h] [rbp-80h] BYREF
  __int64 v40; // [rsp+78h] [rbp-78h]
  unsigned __int64 *v41; // [rsp+80h] [rbp-70h]
  _QWORD *v42; // [rsp+88h] [rbp-68h]
  __int64 v43; // [rsp+90h] [rbp-60h]
  int v44; // [rsp+98h] [rbp-58h]
  __int64 v45; // [rsp+A0h] [rbp-50h]
  __int64 v46; // [rsp+A8h] [rbp-48h]

  v5 = (_QWORD *)sub_16498A0(a3);
  v6 = *(unsigned __int8 **)(a3 + 48);
  v39 = 0;
  v42 = v5;
  v7 = *(_QWORD *)(a3 + 40);
  v43 = 0;
  v40 = v7;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v41 = (unsigned __int64 *)(a3 + 24);
  v37[0] = v6;
  if ( v6 )
  {
    sub_1623A60((__int64)v37, (__int64)v6, 2);
    if ( v39 )
      sub_161E7C0((__int64)&v39, (__int64)v39);
    v39 = v37[0];
    if ( v37[0] )
      sub_1623210((__int64)v37, v37[0], (__int64)&v39);
  }
  v8 = (_BYTE *)a1[13];
  v36 = 257;
  if ( !v8 )
  {
    v8 = *(_BYTE **)(*a1 + 224LL);
    if ( v8 )
      a1[13] = v8;
    else
      v8 = (_BYTE *)sub_18179B0(a1);
  }
  v9 = sub_1643360(v42);
  v33 = (__int64 *)sub_159C470(v9, 0, 0);
  v10 = sub_1643360(v42);
  v11 = sub_159C470(v10, a2, 0);
  v12 = v8[16] <= 0x10u;
  v34 = v11;
  if ( v12 )
  {
    BYTE4(v37[0]) = 0;
    v13 = sub_15A2E80(0, (__int64)v8, &v33, 2u, 0, (__int64)v37, 0);
    v14 = v39;
    v15 = (_QWORD *)v13;
LABEL_9:
    if ( v14 )
      sub_161E7C0((__int64)&v39, (__int64)v14);
    return v15;
  }
  v38 = 257;
  v17 = *(_QWORD *)v8;
  if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16 )
    v17 = **(_QWORD **)(v17 + 16);
  v18 = *(_QWORD *)(v17 + 24);
  v19 = sub_1648A60(72, 3u);
  v15 = v19;
  if ( v19 )
  {
    v31 = (__int64)v19;
    v20 = (__int64)(v19 - 9);
    v21 = *(_QWORD *)v8;
    if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16 )
      v21 = **(_QWORD **)(v21 + 16);
    v30 = *(_DWORD *)(v21 + 8) >> 8;
    v22 = (__int64 *)sub_15F9F50(v18, (__int64)&v33, 2);
    v23 = (__int64 *)sub_1646BA0(v22, v30);
    v24 = *(_QWORD *)v8;
    if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16
      || (v24 = *v33, *(_BYTE *)(*v33 + 8) == 16)
      || (v24 = *(_QWORD *)v34, *(_BYTE *)(*(_QWORD *)v34 + 8LL) == 16) )
    {
      v23 = sub_16463B0(v23, *(_QWORD *)(v24 + 32));
    }
    sub_15F1EA0((__int64)v15, (__int64)v23, 32, v20, 3, 0);
    v15[7] = v18;
    v15[8] = sub_15F9F50(v18, (__int64)&v33, 2);
    sub_15F9CE0((__int64)v15, (__int64)v8, (__int64 *)&v33, 2, (__int64)v37);
  }
  else
  {
    v31 = 0;
  }
  if ( v40 )
  {
    v25 = v41;
    sub_157E9D0(v40 + 40, (__int64)v15);
    v26 = v15[3];
    v27 = *v25;
    v15[4] = v25;
    v27 &= 0xFFFFFFFFFFFFFFF8LL;
    v15[3] = v27 | v26 & 7;
    *(_QWORD *)(v27 + 8) = v15 + 3;
    *v25 = *v25 & 7 | (unsigned __int64)(v15 + 3);
  }
  sub_164B780(v31, &v35);
  if ( v39 )
  {
    v32 = v39;
    sub_1623A60((__int64)&v32, (__int64)v39, 2);
    v28 = v15[6];
    if ( v28 )
      sub_161E7C0((__int64)(v15 + 6), v28);
    v29 = v32;
    v15[6] = v32;
    if ( v29 )
      sub_1623210((__int64)&v32, v29, (__int64)(v15 + 6));
    v14 = v39;
    goto LABEL_9;
  }
  return v15;
}
