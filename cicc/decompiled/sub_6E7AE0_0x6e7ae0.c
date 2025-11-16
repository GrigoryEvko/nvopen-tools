// Function: sub_6E7AE0
// Address: 0x6e7ae0
//
__int64 __fastcall sub_6E7AE0(__int64 *a1, __int64 a2, _BOOL4 a3, unsigned int a4, int a5, char a6, int a7, _DWORD *a8)
{
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 result; // rax
  _DWORD *v16; // rax
  _DWORD *v17; // rsi
  _DWORD *v18; // r12
  __int64 v19; // rdi
  __int64 v20; // r13
  char v21; // al
  _QWORD *v22; // r15
  __int64 v23; // r12
  __int64 v24; // rax
  char v25; // al
  __int64 v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r15
  char i; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // r12
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // r15
  char v43; // al
  int v44; // eax
  int v45; // eax
  int v46; // eax
  int v47; // eax
  __int64 v48; // [rsp+0h] [rbp-60h]
  _QWORD *v49; // [rsp+0h] [rbp-60h]
  __int64 v50; // [rsp+8h] [rbp-58h]
  __int64 v51; // [rsp+8h] [rbp-58h]
  _QWORD *v52; // [rsp+8h] [rbp-58h]
  _QWORD *v53; // [rsp+8h] [rbp-58h]
  bool v55; // [rsp+10h] [rbp-50h]
  __int64 v56; // [rsp+10h] [rbp-50h]
  __int64 v57; // [rsp+10h] [rbp-50h]
  _QWORD *v58; // [rsp+10h] [rbp-50h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  __int64 v60; // [rsp+18h] [rbp-48h]
  int v61; // [rsp+24h] [rbp-3Ch] BYREF
  _DWORD *v62[7]; // [rsp+28h] [rbp-38h] BYREF

  v13 = (__int64 *)*a1;
  v13[2] = 0;
  v59 = *v13;
  if ( !(unsigned int)sub_6E6010() )
    a3 = 0;
  if ( dword_4F077C4 != 2 )
    goto LABEL_4;
  if ( a7 )
    goto LABEL_5;
  if ( (unsigned int)sub_8D2F30(v59, a2) && (unsigned int)sub_8D5EF0(v59, a2, &v61, v62) )
  {
    v16 = (_DWORD *)sub_8D46C0(a2);
    if ( v61 )
      return sub_6E7420((__int64)v62[0], v16, a3, a4, 0, a5, 0, a1, a8, 0);
    else
      return (__int64)sub_6E7880((__int64)v16, v62[0], a4, 0, a1, a8, 0);
  }
  if ( dword_4F077C4 != 2 )
  {
LABEL_4:
    if ( unk_4F07778 <= 199900 )
    {
      if ( !dword_4F077C0 || !(unsigned int)sub_8D29A0(a2) )
        goto LABEL_6;
      goto LABEL_11;
    }
LABEL_5:
    if ( !(unsigned int)sub_8D29A0(a2) )
    {
LABEL_6:
      v14 = sub_73DBF0(5, a2, *a1);
      *a1 = v14;
      *(_BYTE *)(v14 + 58) = (2 * (a6 & 1)) | *(_BYTE *)(v14 + 58) & 0xFD;
      result = (4 * (a7 & 1)) | *(_BYTE *)(*a1 + 58) & 0xFBu;
      *(_BYTE *)(*a1 + 58) = (4 * (a7 & 1)) | *(_BYTE *)(*a1 + 58) & 0xFB;
      if ( a5 )
      {
        *(_BYTE *)(*a1 + 27) |= 2u;
        result = *a1;
        *(_QWORD *)(*a1 + 28) = *(_QWORD *)a8;
      }
      return result;
    }
LABEL_11:
    result = sub_73DBF0(20, a2, *a1);
    *a1 = result;
    if ( a5 )
    {
      *(_BYTE *)(result + 27) |= 2u;
      result = *a1;
      *(_QWORD *)(*a1 + 28) = *(_QWORD *)a8;
    }
    return result;
  }
  if ( !(unsigned int)sub_8D3D10(v59) || !(unsigned int)sub_8D3D10(a2) || !(unsigned int)sub_8D5F90(v59, a2, &v61, v62) )
  {
    if ( dword_4F077C4 == 2 )
      goto LABEL_5;
    goto LABEL_4;
  }
  v17 = (_DWORD *)a4;
  v18 = v62[0];
  v55 = a4 != 0;
  if ( v61 )
  {
    v19 = *(_QWORD *)*a1;
    v20 = sub_8D4870(v19);
    v21 = *((_BYTE *)v18 + 96);
    if ( (v21 & 4) != 0 && v55 )
    {
      if ( (unsigned int)sub_6E5430() )
      {
        v17 = a8;
        v19 = 286;
        sub_685360(0x11Eu, a8, *((_QWORD *)v18 + 5));
      }
      goto LABEL_31;
    }
    if ( (v21 & 2) != 0 )
    {
      if ( !qword_4D0495C )
      {
LABEL_30:
        v23 = *((_QWORD *)v18 + 5);
        v24 = sub_8D4890(*(_QWORD *)*a1);
        v17 = a8;
        v19 = 916;
        sub_6E5ED0(0x394u, a8, v24, v23);
LABEL_31:
        result = sub_7305B0(v19, v17);
        *a1 = result;
        return result;
      }
      result = *((_QWORD *)v18 + 14);
      v22 = *(_QWORD **)(result + 16);
      v38 = v22;
    }
    else
    {
      result = *((_QWORD *)v18 + 14);
      v22 = *(_QWORD **)(result + 8);
      if ( (*(_BYTE *)(v22[2] + 96LL) & 2) != 0 && !qword_4D0495C )
        goto LABEL_30;
      v38 = *(_QWORD **)(result + 16);
    }
    for ( ; (_QWORD *)*v38 != v22; v22 = (_QWORD *)*v22 )
    {
      if ( v22 == v38 )
      {
        result = sub_73DBF0(16, a2, *a1);
        *a1 = result;
      }
      else
      {
        v39 = sub_73F230(v20, *(_QWORD *)(v22[2] + 40LL));
        result = sub_73DBF0(16, v39, *a1);
        *a1 = result;
        *(_BYTE *)(result + 58) |= 0x80u;
      }
    }
    return result;
  }
  v60 = sub_8D4890(a2);
  if ( !(unsigned int)sub_6E6010() )
    a3 = 0;
  v25 = *((_BYTE *)v18 + 96);
  if ( (v25 & 4) != 0 && v55 )
  {
    sub_6E5ED0(0x11Fu, a8, v60, *((_QWORD *)v18 + 5));
    result = sub_7305B0(287, a8);
    *a1 = result;
    return result;
  }
  if ( (v25 & 2) != 0
    || (v26 = *((_QWORD *)v18 + 14), v27 = *(_QWORD **)(v26 + 8), (*(_BYTE *)(v27[2] + 96LL) & 2) != 0) )
  {
    sub_6E5ED0(0x324u, a8, v60, *((_QWORD *)v18 + 5));
    result = sub_7305B0(804, a8);
    *a1 = result;
    return result;
  }
  if ( !a3 )
    goto LABEL_40;
  v58 = *(_QWORD **)(v26 + 16);
  if ( v27 == (_QWORD *)*v58 )
    goto LABEL_40;
  v40 = v60;
  while ( 1 )
  {
    v42 = v27[2];
    v43 = *(_BYTE *)(v42 + 96);
    if ( (v43 & 2) == 0 )
      break;
    if ( (v43 & 1) != 0 )
    {
      v41 = *(_QWORD *)(v42 + 112);
      if ( !*(_QWORD *)v41 )
        goto LABEL_65;
    }
    v52 = v27;
    v44 = sub_87DE40(v27[2], v40);
    v27 = v52;
    if ( !v44 )
      goto LABEL_71;
LABEL_66:
    v40 = *(_QWORD *)(v42 + 40);
    v27 = (_QWORD *)*v27;
    if ( (_QWORD *)*v58 == v27 )
      goto LABEL_77;
  }
  v41 = *(_QWORD *)(v42 + 112);
LABEL_65:
  if ( !*(_BYTE *)(v41 + 25) )
    goto LABEL_66;
  v49 = v27;
  v46 = sub_87D890(v40);
  v27 = v49;
  if ( v46 )
    goto LABEL_66;
  if ( *(_BYTE *)(*(_QWORD *)(v42 + 112) + 25LL) != 1 )
    goto LABEL_75;
  v47 = sub_87D970(v40);
  v27 = v49;
  if ( v47 )
    goto LABEL_66;
LABEL_71:
  if ( *(_BYTE *)(*(_QWORD *)(v42 + 112) + 25LL) == 1 )
  {
    v53 = v27;
    if ( dword_4F077BC )
    {
      if ( qword_4F077A8 <= 0x9DCFu )
      {
        v45 = sub_87E070(v42, v18);
        v27 = v53;
        if ( v45 )
          goto LABEL_66;
      }
    }
  }
LABEL_75:
  if ( sub_6E53E0(7, 0x500u, a8) )
    sub_685260(7u, 0x500u, a8, *(_QWORD *)(v42 + 40));
LABEL_77:
  v26 = *((_QWORD *)v18 + 14);
  v28 = *(_QWORD *)(v26 + 16);
  if ( (v18[24] & 2) == 0 )
LABEL_40:
    v28 = *(_QWORD *)(v26 + 8);
  v29 = *(_QWORD *)(v26 + 16);
  for ( i = 2 * (a5 & 1); *(_QWORD *)(v28 + 8) != v29; v29 = *(_QWORD *)(v29 + 8) )
  {
    v34 = *(_QWORD *)*a1;
    if ( v28 == v29 )
    {
      v51 = v28;
      v35 = sub_8D4870(v34);
      v57 = *a1;
      v36 = sub_73F230(v35, v60);
      v37 = sub_73DBF0(17, v36, v57);
      *a1 = v37;
      *(_BYTE *)(v37 + 27) = i | *(_BYTE *)(v37 + 27) & 0xFD;
      v28 = v51;
    }
    else
    {
      v48 = v28;
      v50 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v29 + 8) + 16LL) + 40LL);
      v31 = sub_8D4870(v34);
      v56 = *a1;
      v32 = sub_73F230(v31, v50);
      v33 = sub_73DBF0(17, v32, v56);
      *a1 = v33;
      *(_BYTE *)(v33 + 27) = i | *(_BYTE *)(v33 + 27) & 0xFD;
      v28 = v48;
      if ( !a5 )
        *(_BYTE *)(*a1 + 58) |= 0x80u;
    }
  }
  result = *a1;
  *(_QWORD *)*a1 = a2;
  return result;
}
