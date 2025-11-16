// Function: sub_1C54710
// Address: 0x1c54710
//
void __fastcall sub_1C54710(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __m128i a6, __m128i a7)
{
  __int64 v7; // r10
  __int16 v12; // ax
  __int64 *v13; // r14
  __int64 v14; // rdi
  bool v15; // al
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // r11
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // eax
  char v27; // cl
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // edx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 *v42; // r14
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 *v49; // r14
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned __int64 v52; // rax
  __int64 *v53; // r12
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 *v56; // r12
  __int64 v57; // rax
  __int64 v58; // [rsp+8h] [rbp-D8h]
  __int64 v59; // [rsp+10h] [rbp-D0h]
  __int64 v60; // [rsp+10h] [rbp-D0h]
  __int64 v61; // [rsp+10h] [rbp-D0h]
  __int64 v62; // [rsp+10h] [rbp-D0h]
  unsigned int v63; // [rsp+18h] [rbp-C8h]
  __int64 v64; // [rsp+18h] [rbp-C8h]
  __int64 v65; // [rsp+18h] [rbp-C8h]
  __int64 v66; // [rsp+20h] [rbp-C0h]
  __int64 v67; // [rsp+20h] [rbp-C0h]
  __int64 v68; // [rsp+20h] [rbp-C0h]
  __int64 v69; // [rsp+20h] [rbp-C0h]
  __int64 *v70; // [rsp+28h] [rbp-B8h]
  __int64 v71; // [rsp+28h] [rbp-B8h]
  __int64 v72; // [rsp+28h] [rbp-B8h]
  __int64 v73; // [rsp+28h] [rbp-B8h]
  __int64 v74; // [rsp+28h] [rbp-B8h]
  __int64 v75; // [rsp+28h] [rbp-B8h]
  __int64 v76; // [rsp+28h] [rbp-B8h]
  __int64 v77; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v78; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v79; // [rsp+48h] [rbp-98h] BYREF
  __int64 v80; // [rsp+50h] [rbp-90h] BYREF
  __int64 v81; // [rsp+58h] [rbp-88h] BYREF
  _BYTE *v82; // [rsp+60h] [rbp-80h] BYREF
  __int64 v83; // [rsp+68h] [rbp-78h]
  _BYTE v84[112]; // [rsp+70h] [rbp-70h] BYREF

  v7 = a1;
  v12 = *(_WORD *)(a1 + 24);
  if ( v12 == 4 )
  {
LABEL_2:
    v13 = *(__int64 **)(v7 + 32);
    v70 = &v13[*(_QWORD *)(v7 + 40)];
    while ( v70 != v13 )
    {
      v14 = *v13++;
      sub_1C54710(v14, a2, a3, a4, a5);
    }
    return;
  }
  while ( 1 )
  {
    if ( v12 == 7 )
    {
      v71 = v7;
      v15 = sub_14560B0(**(_QWORD **)(v7 + 32));
      v7 = v71;
      if ( v15 )
        goto LABEL_22;
      v66 = *(_QWORD *)(v71 + 48);
      v63 = *(_WORD *)(v71 + 26) & 7;
      v16 = sub_13A5BC0((_QWORD *)v71, a4);
      v17 = sub_1456040(**(_QWORD **)(v71 + 32));
      v18 = sub_145CF80(a4, v17, 0, 0);
      v19 = sub_14799E0(a4, v18, v16, v66, v63);
      sub_1C54710(v19, a2, a3, a4, a5);
      v7 = **(_QWORD **)(v71 + 32);
      goto LABEL_7;
    }
    if ( v12 != 5 )
      break;
    if ( *(_QWORD *)(v7 + 40) != 2 )
      goto LABEL_22;
    v20 = *(__int64 **)(v7 + 32);
    if ( *(_WORD *)(*v20 + 24) )
      goto LABEL_22;
    v72 = v7;
    if ( a2 )
    {
      a2 = sub_13A5B60(a4, a2, *v20, 0, 0);
      v20 = *(__int64 **)(v72 + 32);
    }
    else
    {
      a2 = *v20;
    }
    v7 = v20[1];
LABEL_7:
    v12 = *(_WORD *)(v7 + 24);
    if ( v12 == 4 )
      goto LABEL_2;
  }
  if ( v12 == 3 )
  {
    v21 = *(_QWORD *)(v7 + 32);
    v22 = *(unsigned __int16 *)(v21 + 24);
    if ( ((unsigned __int16)(v22 - 7) <= 2u || (unsigned int)(v22 - 4) <= 1) && (*(_BYTE *)(v21 + 26) & 4) == 0 )
    {
      v76 = v7;
      v69 = *(_QWORD *)(v7 + 32);
      v51 = sub_1456040(v69);
      v52 = sub_1456C90(a4, v51);
      v7 = v76;
      if ( v52 <= 0x1F || (v21 = v69, !byte_4FBC5A0) )
      {
        if ( a2 )
          v7 = sub_13A5B60(a4, a2, v76, 0, 0);
        v82 = (_BYTE *)v7;
        sub_1458920(a3, &v82);
        return;
      }
    }
    v59 = v7;
    v82 = v84;
    v64 = v21;
    v83 = 0x800000000LL;
    v23 = sub_1456040(v21);
    v80 = sub_145CF80(a4, v23, 0, 0);
    v67 = v80;
    sub_1C54710(v64, 0, &v82, a4, &v80);
    v7 = v59;
    if ( v67 == v80 )
    {
LABEL_20:
      if ( v82 != v84 )
      {
        v73 = v7;
        _libc_free((unsigned __int64)v82);
        v7 = v73;
      }
      goto LABEL_22;
    }
    v46 = sub_1456040(v59);
    v47 = sub_147B0D0(a4, v80, v46, 0);
    if ( a2 )
    {
      v48 = sub_13A5B60(a4, a2, v47, 0, 0);
      *a5 = sub_13A5B00(a4, *a5, v48, 0, 0);
      v49 = sub_147DD40(a4, (__int64 *)&v82, 0, 0, a6, a7);
      v50 = sub_1456040(v59);
      v44 = sub_147B0D0(a4, (__int64)v49, v50, 0);
      goto LABEL_45;
    }
    *a5 = sub_13A5B00(a4, *a5, v47, 0, 0);
    v53 = sub_147DD40(a4, (__int64 *)&v82, 0, 0, a6, a7);
    v54 = sub_1456040(v59);
    v45 = sub_147B0D0(a4, (__int64)v53, v54, 0);
LABEL_46:
    v81 = v45;
    sub_1458920(a3, &v81);
    if ( v82 != v84 )
      _libc_free((unsigned __int64)v82);
  }
  else
  {
    if ( v12 )
    {
      if ( !byte_4FBCE60 )
        goto LABEL_22;
      if ( v12 != 2 )
        goto LABEL_22;
      v24 = *(_QWORD *)(v7 + 32);
      v74 = v7;
      v78 = 0;
      v68 = v24;
      v25 = sub_1456040(v24);
      v26 = sub_1456C90(a4, v25);
      v7 = v74;
      v27 = v26;
      if ( v26 > 0x20 )
        goto LABEL_22;
      if ( !byte_4FBC680 || v26 != 32 || (v55 = sub_1456C90(a4, *(_QWORD *)(v74 + 40)), v7 = v74, v27 = 32, v55 != 64) )
      {
        v75 = v7;
        v28 = sub_1C52500(v68, a4, &v77, (1LL << v27) - 1, (unsigned __int64 *)&v78);
        v7 = v75;
        if ( !v28 )
        {
LABEL_22:
          if ( a2 )
            v7 = sub_13A5B60(a4, a2, v7, 0, 0);
          v82 = (_BYTE *)v7;
          sub_1458920(a3, &v82);
          return;
        }
      }
      v60 = v7;
      v82 = v84;
      v83 = 0x800000000LL;
      v29 = sub_1456040(v68);
      v65 = sub_145CF80(a4, v29, 0, 0);
      v79 = v65;
      sub_1C54710(v68, 0, &v82, a4, &v79);
      v7 = v60;
      if ( v65 == v79 )
        goto LABEL_20;
      v30 = *(_QWORD *)(v79 + 32);
      v31 = *(_DWORD *)(v30 + 32);
      v32 = *(__int64 **)(v30 + 24);
      v33 = v31 > 0x40 ? *v32 : (__int64)((_QWORD)v32 << (64 - (unsigned __int8)v31)) >> (64 - (unsigned __int8)v31);
      if ( (int)v78 >= v33 )
        goto LABEL_20;
      v58 = v60;
      v61 = v78;
      v34 = sub_1456040(v68);
      v80 = sub_145CF80(a4, v34, v61, 0);
      v35 = *(_QWORD *)(v79 + 32);
      v36 = *(_DWORD *)(v35 + 32);
      if ( v36 > 0x40 )
        v37 = **(_QWORD **)(v35 + 24);
      else
        v37 = (__int64)(*(_QWORD *)(v35 + 24) << (64 - (unsigned __int8)v36)) >> (64 - (unsigned __int8)v36);
      v62 = v37 - v78;
      v38 = sub_1456040(v68);
      v79 = sub_145CF80(a4, v38, v62, 0);
      v39 = sub_1456040(v58);
      v40 = sub_14747F0(a4, v79, v39, 0);
      if ( a2 )
      {
        v41 = sub_13A5B60(a4, a2, v40, 0, 0);
        *a5 = sub_13A5B00(a4, *a5, v41, 0, 0);
        if ( v65 != v80 )
          sub_1458920((__int64)&v82, &v80);
        v42 = sub_147DD40(a4, (__int64 *)&v82, 0, 0, a6, a7);
        v43 = sub_1456040(v58);
        v44 = sub_14747F0(a4, (__int64)v42, v43, 0);
LABEL_45:
        v45 = sub_13A5B60(a4, a2, v44, 0, 0);
      }
      else
      {
        *a5 = sub_13A5B00(a4, *a5, v40, 0, 0);
        if ( v65 != v80 )
          sub_1458920((__int64)&v82, &v80);
        v56 = sub_147DD40(a4, (__int64 *)&v82, 0, 0, a6, a7);
        v57 = sub_1456040(v58);
        v45 = sub_14747F0(a4, (__int64)v56, v57, 0);
      }
      goto LABEL_46;
    }
    if ( a2 )
      v7 = sub_13A5B60(a4, a2, v7, 0, 0);
    *a5 = sub_13A5B00(a4, *a5, v7, 0, 0);
  }
}
