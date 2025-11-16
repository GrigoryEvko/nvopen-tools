// Function: sub_1286E40
// Address: 0x1286e40
//
__int64 __fastcall sub_1286E40(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // rcx
  _BYTE *v9; // r15
  unsigned int v10; // r8d
  char v11; // al
  __int64 v12; // rax
  char v13; // dl
  char v14; // bl
  unsigned int v15; // eax
  int v16; // eax
  __int64 *v18; // r9
  unsigned __int8 *v19; // rax
  int v20; // edx
  __int64 v21; // r14
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r10
  unsigned int v26; // ecx
  __int64 v27; // rax
  _QWORD *v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rsi
  char v35; // al
  int v36; // r12d
  _BOOL4 v37; // edx
  bool v38; // al
  int v39; // [rsp+Ch] [rbp-84h]
  unsigned int v40; // [rsp+Ch] [rbp-84h]
  unsigned int v41; // [rsp+10h] [rbp-80h]
  unsigned int v42; // [rsp+10h] [rbp-80h]
  __int64 v43; // [rsp+10h] [rbp-80h]
  unsigned __int64 *v44; // [rsp+10h] [rbp-80h]
  __int64 v45; // [rsp+10h] [rbp-80h]
  __int64 v46; // [rsp+18h] [rbp-78h]
  _QWORD *v47; // [rsp+18h] [rbp-78h]
  unsigned int v48; // [rsp+24h] [rbp-6Ch] BYREF
  __int64 v49; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v50[2]; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v51; // [rsp+40h] [rbp-50h]

  v6 = *(_QWORD *)(a3 + 72);
  v7 = *(_QWORD *)(v6 + 16);
  if ( *(_BYTE *)(v6 + 24) != 3 )
    goto LABEL_2;
  if ( !(unsigned __int8)sub_127F7A0(a2[4], *(_QWORD *)(v6 + 56), &v48) )
    goto LABEL_2;
  a4 = v48;
  if ( v48 > 3 )
    goto LABEL_2;
  if ( *(_BYTE *)(v7 + 24) != 4 )
    goto LABEL_2;
  v19 = *(unsigned __int8 **)(*(_QWORD *)(v7 + 56) + 8LL);
  if ( !v19 )
    goto LABEL_2;
  v20 = *v19;
  if ( v20 == 120 )
  {
    a5 = v19[1];
    if ( !v19[1] )
      goto LABEL_17;
  }
  if ( v20 == 121 )
  {
    a5 = 1;
    if ( !v19[1] )
      goto LABEL_17;
  }
  if ( v20 != 122 || v19[1] )
  {
LABEL_2:
    sub_1286D80((__int64)v50, a2, v6, a4, a5);
    v8 = *(_QWORD *)v6;
    v9 = (_BYTE *)v50[1];
    v10 = v51;
    v11 = *(_BYTE *)(*(_QWORD *)v6 + 140LL);
    if ( v11 == 12 )
    {
      v12 = *(_QWORD *)v6;
      do
      {
        v12 = *(_QWORD *)(v12 + 160);
        v13 = *(_BYTE *)(v12 + 140);
      }
      while ( v13 == 12 );
      v14 = v13 == 11;
    }
    else
    {
      v14 = v11 == 11;
      if ( (v11 & 0xFB) != 8 )
      {
        LOBYTE(v16) = 0;
        goto LABEL_9;
      }
    }
    v41 = v51;
    v46 = v8;
    v15 = sub_8D4C10(v8, dword_4F077C4 != 2);
    v8 = v46;
    v10 = v41;
    v16 = (v15 >> 1) & 1;
LABEL_9:
    sub_1280460(a1, a2, v9, v8, v10, v7, v14, v16);
    return a1;
  }
  LODWORD(a5) = 2;
LABEL_17:
  v21 = *v18;
  v42 = v48;
  v39 = a5;
  v22 = *v18;
  v50[0] = "predef_tmp_comp";
  LOWORD(v51) = 259;
  v47 = sub_127FDC0(a2, v22, (__int64)v50);
  LOWORD(v51) = 257;
  v23 = sub_126A190((_QWORD *)a2[4], dword_427F760[3 * v42 + v39], 0, 0);
  v24 = sub_1285290(a2 + 6, *(_QWORD *)(*(_QWORD *)v23 + 24LL), v23, 0, 0, (__int64)v50, 0);
  v25 = v24;
  v26 = unk_4D0463C;
  if ( unk_4D0463C )
  {
    v45 = v24;
    v38 = sub_126A420(a2[4], (unsigned __int64)v47);
    v25 = v45;
    v26 = v38;
  }
  v40 = v26;
  v43 = v25;
  LOWORD(v51) = 257;
  v27 = sub_1648A60(64, 2);
  v28 = (_QWORD *)v27;
  if ( v27 )
    sub_15F9650(v27, v43, v47, v40, 0);
  v29 = a2[7];
  if ( v29 )
  {
    v44 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v29 + 40, v28);
    v30 = *v44;
    v31 = v28[3] & 7LL;
    v28[4] = v44;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    v28[3] = v30 | v31;
    *(_QWORD *)(v30 + 8) = v28 + 3;
    *v44 = *v44 & 7 | (unsigned __int64)(v28 + 3);
  }
  sub_164B780(v28, v50);
  v32 = a2[6];
  if ( v32 )
  {
    v49 = a2[6];
    sub_1623A60(&v49, v32, 2);
    if ( v28[6] )
      sub_161E7C0(v28 + 6);
    v33 = v49;
    v28[6] = v49;
    if ( v33 )
      sub_1623210(&v49, v33, v28 + 6);
  }
  if ( *(char *)(v21 + 142) >= 0 && *(_BYTE *)(v21 + 140) == 12 )
    v34 = (unsigned int)sub_8D4AB0(v21);
  else
    v34 = *(unsigned int *)(v21 + 136);
  sub_15F9450(v28, v34);
  if ( *(char *)(v21 + 142) < 0 )
  {
    v36 = *(_DWORD *)(v21 + 136);
    v35 = *(_BYTE *)(v21 + 140);
  }
  else
  {
    v35 = *(_BYTE *)(v21 + 140);
    if ( v35 == 12 )
    {
      v36 = sub_8D4AB0(v21);
      v35 = *(_BYTE *)(v21 + 140);
    }
    else
    {
      v36 = *(_DWORD *)(v21 + 136);
    }
  }
  v37 = 0;
  if ( (v35 & 0xFB) == 8 )
    v37 = (sub_8D4C10(v21, dword_4F077C4 != 2) & 2) != 0;
  *(_DWORD *)a1 = 0;
  *(_DWORD *)(a1 + 40) = v37;
  *(_QWORD *)(a1 + 8) = v47;
  *(_DWORD *)(a1 + 16) = v36;
  return a1;
}
