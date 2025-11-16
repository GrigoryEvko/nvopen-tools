// Function: sub_29A5A80
// Address: 0x29a5a80
//
__int64 __fastcall sub_29A5A80(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rax
  bool v9; // cc
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // r15
  unsigned __int64 v13; // rdi
  _QWORD *v14; // rax
  _QWORD *v15; // r9
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  _QWORD *v20; // rax
  _QWORD *v21; // r8
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rsi
  int v25; // edx
  __int64 v26; // rax
  _QWORD *v27; // rcx
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rax
  _QWORD *v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdi
  __int16 v34; // dx
  unsigned __int64 *v35; // r10
  unsigned __int8 v36; // al
  char v37; // dl
  __int64 v38; // rcx
  __int16 v39; // dx
  unsigned __int64 *v40; // r9
  unsigned __int8 v41; // al
  char v42; // dl
  __int64 v43; // rcx
  _BYTE *v45; // [rsp+8h] [rbp-98h]
  _QWORD *v46; // [rsp+10h] [rbp-90h]
  _QWORD *v47; // [rsp+10h] [rbp-90h]
  _QWORD *v48; // [rsp+18h] [rbp-88h]
  _QWORD *v49; // [rsp+18h] [rbp-88h]
  __int64 v50; // [rsp+18h] [rbp-88h]
  _QWORD *v51; // [rsp+18h] [rbp-88h]
  unsigned int v52; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v53; // [rsp+24h] [rbp-7Ch] BYREF
  unsigned int v54; // [rsp+28h] [rbp-78h] BYREF
  unsigned int v55; // [rsp+2Ch] [rbp-74h] BYREF
  _QWORD *v56; // [rsp+30h] [rbp-70h] BYREF
  __int64 v57; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v58[12]; // [rsp+40h] [rbp-60h] BYREF

  if ( !sub_30A7C70(a3) )
    return 0;
  v6 = sub_B43CB0(a1);
  v7 = sub_30A7D00(a1);
  if ( !v7 )
    return 0;
  v48 = (_QWORD *)v7;
  v8 = sub_B59BC0(v7);
  v9 = *(_DWORD *)(v8 + 32) <= 0x40u;
  v10 = *(_QWORD **)(v8 + 24);
  if ( !v9 )
    v10 = (_QWORD *)*v10;
  v56 = v10;
  v11 = sub_29A5610(a1, (__int64)a2, 0);
  v12 = sub_29A3E20(v11, a2, 0);
  sub_B444E0(v48, a1 + 24, 0);
  v13 = sub_30A7C70(a3);
  v14 = *(_QWORD **)(a3 + 112);
  v15 = (_QWORD *)(a3 + 104);
  if ( v14 )
  {
    do
    {
      while ( 1 )
      {
        v16 = v14[2];
        v17 = v14[3];
        if ( v13 <= v14[4] )
          break;
        v14 = (_QWORD *)v14[3];
        if ( !v17 )
          goto LABEL_10;
      }
      v15 = v14;
      v14 = (_QWORD *)v14[2];
    }
    while ( v16 );
LABEL_10:
    if ( (_QWORD *)(a3 + 104) != v15 && v13 < v15[4] )
      v15 = (_QWORD *)(a3 + 104);
  }
  v46 = (_QWORD *)(a3 + 104);
  v52 = *((_DWORD *)v15 + 11);
  *((_DWORD *)v15 + 11) = v52 + 1;
  v49 = (_QWORD *)sub_B47F80(v48);
  sub_B59C10((__int64)v49, v52);
  sub_B59D40((__int64)v49, (__int64)a2);
  sub_B44220(v49, v12 + 24, 0);
  v18 = *(_QWORD *)(a1 + 40);
  v50 = *(_QWORD *)(v12 + 40);
  v19 = sub_30A7C70(a3);
  v20 = *(_QWORD **)(a3 + 112);
  v21 = (_QWORD *)(a3 + 104);
  if ( v20 )
  {
    do
    {
      while ( 1 )
      {
        v22 = v20[2];
        v23 = v20[3];
        if ( v19 <= v20[4] )
          break;
        v20 = (_QWORD *)v20[3];
        if ( !v23 )
          goto LABEL_18;
      }
      v21 = v20;
      v20 = (_QWORD *)v20[2];
    }
    while ( v22 );
LABEL_18:
    if ( v46 != v21 && v19 < v21[4] )
      v21 = (_QWORD *)(a3 + 104);
  }
  v24 = v6;
  v25 = *((_DWORD *)v21 + 10) + 1;
  v53 = *((_DWORD *)v21 + 10);
  *((_DWORD *)v21 + 10) = v25;
  v26 = sub_30A7C70(a3);
  v27 = (_QWORD *)(a3 + 104);
  v28 = v26;
  v29 = *(_QWORD **)(a3 + 112);
  if ( v29 )
  {
    v30 = (_QWORD *)(a3 + 104);
    do
    {
      while ( 1 )
      {
        v24 = v29[2];
        v31 = v29[3];
        if ( v28 <= v29[4] )
          break;
        v29 = (_QWORD *)v29[3];
        if ( !v31 )
          goto LABEL_26;
      }
      v30 = v29;
      v29 = (_QWORD *)v29[2];
    }
    while ( v24 );
LABEL_26:
    if ( v46 != v30 && v28 >= v30[4] )
      v27 = v30;
  }
  v32 = (unsigned int)(*((_DWORD *)v27 + 10) + 1);
  v54 = *((_DWORD *)v27 + 10);
  *((_DWORD *)v27 + 10) = v32;
  v33 = *(_QWORD *)(v6 + 80);
  if ( v33 )
    v33 -= 24;
  v45 = (_BYTE *)sub_30A7DC0(v33, v24, v32, v27);
  v47 = (_QWORD *)sub_B47F80(v45);
  sub_B59C10((__int64)v47, v53);
  v35 = (unsigned __int64 *)sub_AA5190(v50);
  if ( v35 )
  {
    v36 = v34;
    v37 = HIBYTE(v34);
  }
  else
  {
    v37 = 0;
    v36 = 0;
  }
  v38 = v36;
  BYTE1(v38) = v37;
  sub_B44240(v47, v50, v35, v38);
  v51 = (_QWORD *)sub_B47F80(v45);
  sub_B59C10((__int64)v51, v54);
  v40 = (unsigned __int64 *)sub_AA5190(v18);
  if ( v40 )
  {
    v41 = v39;
    v42 = HIBYTE(v39);
  }
  else
  {
    v42 = 0;
    v41 = 0;
  }
  v43 = v41;
  BYTE1(v43) = v42;
  sub_B44240(v51, v18, v40, v43);
  v57 = sub_30A7A60(a2);
  v55 = v54 + 1;
  v58[0] = &v55;
  v58[1] = &v56;
  v58[2] = &v57;
  v58[3] = &v52;
  v58[4] = &v53;
  v58[5] = &v54;
  sub_30A7E90(a3, sub_29A67E0, v58, v6);
  return v12;
}
