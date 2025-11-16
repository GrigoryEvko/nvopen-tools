// Function: sub_12DE330
// Address: 0x12de330
//
__int64 __fastcall sub_12DE330(_BYTE *a1, _BYTE *a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // rdi
  _BYTE *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rax
  _BYTE *v42; // rdi
  _BYTE *v43; // rsi
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v48; // rsi
  __int64 v49; // rsi
  __int64 v50; // rsi
  __int64 v51; // rsi
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // [rsp-10h] [rbp-50h]
  __int64 v55; // [rsp-10h] [rbp-50h]
  __int64 v56; // [rsp-10h] [rbp-50h]
  __int64 v57; // [rsp-10h] [rbp-50h]
  __int64 v58; // [rsp-8h] [rbp-48h]
  _BYTE v59[16]; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v60)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-30h]

  v2 = (__int64)a1;
  v4 = sub_1654860(1);
  sub_12DE0B0((__int64)a1, v4, 0, 0);
  v60 = 0;
  v5 = (_BYTE *)sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)v59);
  sub_12DE0B0((__int64)a1, (__int64)v5, 1u, 0);
  v6 = v54;
  if ( v60 )
  {
    v5 = v59;
    a1 = v59;
    v60(v59, v59, 3);
  }
  v7 = sub_1B26330(a1, v5, v6);
  sub_12DE0B0(v2, v7, 1u, 0);
  v8 = sub_185D600();
  sub_12DE0B0(v2, v8, 1u, 0);
  v9 = sub_1C6E800();
  sub_12DE0B0(v2, v9, 1u, 0);
  v10 = sub_1C6E560();
  sub_12DE0B0(v2, v10, 1u, 0);
  v11 = sub_1857160();
  sub_12DE0B0(v2, v11, 1u, 0);
  v12 = sub_1842BC0();
  sub_12DE0B0(v2, v12, 1u, 0);
  if ( a2[3160] )
  {
    v48 = sub_17060B0(1, 0);
    sub_12DE0B0(v2, v48, 1u, 0);
  }
  v13 = sub_12D4560();
  sub_12DE0B0(v2, v13, 1u, 0);
  v14 = sub_18A3090();
  sub_12DE0B0(v2, v14, 1u, 0);
  v15 = sub_184CD60();
  sub_12DE0B0(v2, v15, 1u, 0);
  if ( !a2[1040] )
  {
    v53 = sub_1869C50(1, 0, 1);
    sub_12DE0B0(v2, v53, 1u, 0);
  }
  v16 = sub_1833EB0(3);
  sub_12DE0B0(v2, v16, 1u, 0);
  v17 = sub_17060B0(1, 0);
  sub_12DE0B0(v2, v17, 1u, 0);
  v18 = sub_1952F90(0xFFFFFFFFLL);
  sub_12DE0B0(v2, v18, 1u, 0);
  v60 = 0;
  v19 = sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)v59);
  sub_12DE0B0(v2, v19, 1u, 0);
  if ( v60 )
    v60(v59, v59, 3);
  v20 = sub_1A223D0();
  sub_12DE0B0(v2, v20, 1u, 0);
  v21 = sub_17060B0(1, 0);
  sub_12DE0B0(v2, v21, 1u, 0);
  v22 = sub_1A7A9F0();
  sub_12DE0B0(v2, v22, 1u, 0);
  v60 = 0;
  v23 = (_BYTE *)v2;
  v24 = (_BYTE *)sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)v59);
  sub_12DE0B0(v2, (__int64)v24, 1u, 0);
  v27 = v55;
  v28 = v58;
  if ( v60 )
  {
    v24 = v59;
    v23 = v59;
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64))v60)(v59, v59, 3, v26, v55, v58);
  }
  v29 = sub_1A02540(v23, v24, v25, v26, v27, v28);
  sub_12DE0B0(v2, v29, 1u, 0);
  v30 = sub_198DF00(0xFFFFFFFFLL);
  sub_12DE0B0(v2, v30, 1u, 0);
  if ( !a2[1320] )
  {
    v52 = sub_1C76260();
    sub_12DE0B0(v2, v52, 1u, 0);
  }
  if ( a2[2880] )
  {
    v51 = sub_195E880(0);
    sub_12DE0B0(v2, v51, 1u, 0);
  }
  if ( !a2[1360] )
  {
    v50 = sub_19C1680(0, 1);
    sub_12DE0B0(v2, v50, 1u, 0);
  }
  if ( a2[3160] )
  {
    v49 = sub_17060B0(1, 0);
    sub_12DE0B0(v2, v49, 1u, 0);
  }
  v31 = sub_19401A0();
  sub_12DE0B0(v2, v31, 1u, 0);
  v32 = sub_1968390();
  sub_12DE0B0(v2, v32, 1u, 0);
  v33 = sub_196A2B0();
  sub_12DE0B0(v2, v33, 1u, 0);
  v34 = sub_19B73C0(2, -1, -1, -1, -1, -1, -1);
  sub_12DE0B0(v2, v34, 1u, 0);
  v35 = sub_17060B0(1, 0);
  sub_12DE0B0(v2, v35, 1u, 0);
  v36 = sub_190BB10(0, 0);
  sub_12DE0B0(v2, v36, 1u, 0);
  v37 = sub_1A13320();
  sub_12DE0B0(v2, v37, 1u, 0);
  v38 = v56;
  v39 = v58;
  if ( a2[3160] )
  {
    v39 = v2;
    v38 = sub_17060B0(1, 1);
    sub_12DE0B0(v2, v38, 1u, 0);
  }
  v40 = sub_18F5480(v39, v38);
  sub_12DE0B0(v2, v40, 1u, 0);
  v41 = sub_18DEFF0();
  sub_12DE0B0(v2, v41, 1u, 0);
  v60 = 0;
  v42 = (_BYTE *)v2;
  v43 = (_BYTE *)sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)v59);
  sub_12DE0B0(v2, (__int64)v43, 1u, 0);
  v44 = v57;
  if ( v60 )
  {
    v43 = v59;
    v42 = v59;
    v60(v59, v59, 3);
  }
  v45 = sub_18B1DE0(v42, v43, v44);
  sub_12DE0B0(v2, v45, 1u, 0);
  v46 = sub_1841180();
  return sub_12DE0B0(v2, v46, 1u, 0);
}
