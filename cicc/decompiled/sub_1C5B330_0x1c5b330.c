// Function: sub_1C5B330
// Address: 0x1c5b330
//
__int64 __fastcall sub_1C5B330(__int64 a1, __int64 *a2, __int64 *a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rbx
  __int64 *v7; // rax
  unsigned __int8 *v8; // rsi
  int v9; // edx
  char v10; // al
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // rax
  unsigned __int8 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int8 *v21; // r10
  __int64 v22; // rax
  __int64 v23; // r10
  __int64 v24; // rax
  bool v25; // al
  __int64 v26; // r10
  bool v27; // al
  __int64 v28; // r10
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rax
  unsigned int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r10
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rbx
  __int64 *v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // [rsp+0h] [rbp-F0h]
  __int64 v54; // [rsp+0h] [rbp-F0h]
  __int64 v55; // [rsp+8h] [rbp-E8h]
  __int64 v56; // [rsp+8h] [rbp-E8h]
  __int64 v57; // [rsp+8h] [rbp-E8h]
  __int64 v58; // [rsp+8h] [rbp-E8h]
  __int64 v59; // [rsp+8h] [rbp-E8h]
  __int64 v60; // [rsp+8h] [rbp-E8h]
  __int64 v61; // [rsp+8h] [rbp-E8h]
  __int64 v62; // [rsp+8h] [rbp-E8h]
  unsigned __int8 *v63; // [rsp+10h] [rbp-E0h]
  __int64 v64; // [rsp+10h] [rbp-E0h]
  __int64 v65; // [rsp+18h] [rbp-D8h]
  __int64 v66; // [rsp+20h] [rbp-D0h]
  unsigned __int8 *v68; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v69; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v70; // [rsp+48h] [rbp-A8h]
  char *v71; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v72; // [rsp+58h] [rbp-98h]
  char v73; // [rsp+60h] [rbp-90h]
  char v74; // [rsp+61h] [rbp-8Fh]
  __int64 *v75[5]; // [rsp+70h] [rbp-80h] BYREF
  int v76; // [rsp+98h] [rbp-58h]
  __int64 v77; // [rsp+A0h] [rbp-50h]
  __int64 v78; // [rsp+A8h] [rbp-48h]

  v6 = *a2;
  v65 = a2[1];
  v66 = a1 + 112;
  if ( (unsigned __int8)sub_1C56330(a1 + 112, a2, v75)
    && v75[0] != (__int64 *)(*(_QWORD *)(a1 + 120) + 32LL * *(unsigned int *)(a1 + 136)) )
  {
    v11 = v75[0][2];
    v12 = v75[0][3];
    if ( v11
      && (*(_BYTE *)(v11 + 16) <= 0x17u || *(_BYTE *)(v6 + 16) <= 0x17u || sub_15CCEE0(*(_QWORD *)(a1 + 200), v11, v6)) )
    {
      *a3 = v12;
      return v11;
    }
    return 0;
  }
  if ( *(_BYTE *)(v6 + 16) <= 0x17u )
    return 0;
  v7 = (__int64 *)sub_16498A0(v6);
  v75[0] = 0;
  v75[3] = v7;
  v75[4] = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v75[1] = *(__int64 **)(v6 + 40);
  v75[2] = (__int64 *)(v6 + 24);
  v8 = *(unsigned __int8 **)(v6 + 48);
  v71 = (char *)v8;
  if ( v8 )
  {
    sub_1623A60((__int64)&v71, (__int64)v8, 2);
    if ( v75[0] )
      sub_161E7C0((__int64)v75, (__int64)v75[0]);
    v75[0] = (__int64 *)v71;
    if ( v71 )
      sub_1623210((__int64)&v71, (unsigned __int8 *)v71, (__int64)v75);
  }
  v9 = *(unsigned __int8 *)(v6 + 16);
  v10 = *(_BYTE *)(v6 + 16);
  if ( (unsigned int)(v9 - 60) > 0xC )
  {
    v11 = 0;
    if ( (unsigned int)(v9 - 35) > 0x11 )
      goto LABEL_10;
    v11 = *(_QWORD *)(v6 - 48);
    v63 = *(unsigned __int8 **)(v6 - 24);
    if ( v10 != 35 )
    {
      if ( dword_4FBC3E0 <= 2 || v10 != 39 )
      {
        v11 = 0;
        goto LABEL_10;
      }
      if ( v63[16] == 13 )
      {
        v62 = sub_146F1B0(*(_QWORD *)(a1 + 184), (__int64)v63);
        if ( !sub_14560B0(v62) )
        {
          v32 = *(_QWORD *)(v62 + 32);
          v33 = *(_DWORD *)(v32 + 32);
          v34 = *(__int64 **)(v32 + 24);
          v35 = v33 > 0x40 ? *v34 : (__int64)((_QWORD)v34 << (64 - (unsigned __int8)v33)) >> (64 - (unsigned __int8)v33);
          v36 = *(_QWORD *)(v65 + 32);
          v37 = *(_DWORD *)(v36 + 32);
          v38 = *(__int64 **)(v36 + 24);
          v39 = v37 > 0x40 ? *v38 : (__int64)((_QWORD)v38 << (64 - (unsigned __int8)v37)) >> (64 - (unsigned __int8)v37);
          v40 = v39 / v35;
          if ( !(v39 % v35)
            && *(_BYTE *)(v11 + 16) > 0x17u
            && (*(_QWORD *)(v11 + 40) == *(_QWORD *)(v6 + 40)
             || (v41 = *(_QWORD *)(v11 + 8)) != 0 && !*(_QWORD *)(v41 + 8)) )
          {
            v54 = v40;
            v42 = *(_QWORD *)(a1 + 184);
            v43 = sub_1456040(v65);
            v69 = v11;
            v68 = 0;
            v70 = sub_145CF80(v42, v43, v54, 0);
            v44 = sub_1C5B330(a1, &v69, &v68);
            v45 = v62;
            v11 = v44;
            if ( v44 )
            {
              v74 = 1;
              v71 = "newMul";
              v73 = 3;
              v46 = sub_156D130((__int64 *)v75, v44, (__int64)v63, (__int64)&v71, 0, 0);
              v45 = v62;
              v11 = v46;
            }
            if ( v68 )
              *a3 = sub_13A5B60(*(_QWORD *)(a1 + 184), v45, (__int64)v68, 0, 0);
            goto LABEL_31;
          }
        }
      }
LABEL_33:
      v11 = 0;
      goto LABEL_31;
    }
    v17 = sub_146F1B0(*(_QWORD *)(a1 + 184), v11);
    v18 = (unsigned __int8 *)v17;
    if ( *(_WORD *)(v17 + 24) )
    {
      sub_1C54F70((__int64 *)&v71, v17, *(_QWORD *)(a1 + 184), a4, a5);
      v18 = (unsigned __int8 *)v71;
    }
    v55 = sub_1456040((__int64)v18);
    if ( v55 != sub_1456040(v65) )
    {
      v56 = *(_QWORD *)(a1 + 184);
      v19 = sub_1456040(v65);
      v18 = (unsigned __int8 *)sub_147B0D0(v56, (__int64)v18, v19, 0);
    }
    if ( (unsigned __int8 *)v65 == v18 )
    {
      if ( *(_BYTE *)(v11 + 16) != 13 )
      {
        v69 = v11;
        v70 = v65;
        v50 = sub_1C5B330(a1, &v69, a3);
        if ( v50 )
        {
          v74 = 1;
          v71 = "newAdd";
          v73 = 3;
          v63 = (unsigned __int8 *)sub_12899C0((__int64 *)v75, v50, (__int64)v63, (__int64)&v71, 0, 0);
        }
        else
        {
          v63 = 0;
        }
      }
      v48 = *a3;
      v49 = sub_1C5B220(v66, a2);
      v49[3] = v48;
      v49[2] = (__int64)v63;
      v11 = (__int64)v63;
      goto LABEL_10;
    }
    v20 = sub_146F1B0(*(_QWORD *)(a1 + 184), (__int64)v63);
    v21 = (unsigned __int8 *)v20;
    if ( *(_WORD *)(v20 + 24) )
    {
      sub_1C54F70((__int64 *)&v71, v20, *(_QWORD *)(a1 + 184), a4, a5);
      v21 = (unsigned __int8 *)v71;
    }
    v53 = (__int64)v21;
    v57 = sub_1456040((__int64)v21);
    v22 = sub_1456040(v65);
    v23 = v53;
    if ( v57 != v22 )
    {
      v58 = *(_QWORD *)(a1 + 184);
      v24 = sub_1456040(v65);
      v23 = sub_147B0D0(v58, v53, v24, 0);
    }
    if ( v65 == v23 )
    {
      if ( v63[16] == 13 )
        goto LABEL_31;
      v69 = (__int64)v63;
      v70 = v65;
      v47 = sub_1C5B330(a1, &v69, a3);
      if ( v47 )
      {
        v74 = 1;
        v71 = "newAdd";
        v73 = 3;
        v11 = sub_12899C0((__int64 *)v75, v11, v47, (__int64)&v71, 0, 0);
        goto LABEL_31;
      }
      goto LABEL_33;
    }
    v59 = v23;
    v68 = 0;
    v69 = 0;
    v25 = sub_14560B0((__int64)v18);
    v26 = v59;
    if ( v25 || (v71 = (char *)v11, v72 = (__int64)v18, v52 = sub_1C5B330(a1, &v71, &v68), v26 = v59, !v52) )
      v68 = v18;
    else
      v11 = v52;
    v60 = v26;
    v27 = sub_14560B0(v26);
    v28 = v60;
    if ( v27 )
    {
      v69 = v60;
      if ( v18 != v68 )
      {
LABEL_48:
        v61 = v28;
        v74 = 1;
        v71 = "newAdd";
        v73 = 3;
        v11 = sub_12899C0((__int64 *)v75, v11, (__int64)v63, (__int64)&v71, 0, 0);
        v64 = *(_QWORD *)(a1 + 184);
        v29 = sub_13A5B00(v64, (__int64)v18, v61, 0, 0);
        v30 = sub_14806B0(v64, v65, v29, 0, 0);
        v31 = (__int64)v68;
        v15 = v30;
        *a3 = v30;
        if ( v31 )
        {
          v15 = sub_13A5B00(*(_QWORD *)(a1 + 184), v30, v31, 0, 0);
          *a3 = v15;
        }
        if ( v69 )
        {
          v15 = sub_13A5B00(*(_QWORD *)(a1 + 184), v15, v69, 0, 0);
          *a3 = v15;
        }
        goto LABEL_32;
      }
    }
    else
    {
      v72 = v60;
      v71 = (char *)v63;
      v51 = sub_1C5B330(a1, &v71, &v69);
      v28 = v60;
      if ( v51 )
        v63 = (unsigned __int8 *)v51;
      else
        v69 = v60;
      if ( v18 != v68 || v60 != v69 )
        goto LABEL_48;
    }
    v11 = 0;
    v15 = *a3;
    goto LABEL_32;
  }
  v11 = 0;
  if ( v10 == 60 )
    goto LABEL_10;
  v14 = *(_QWORD *)(v6 - 24);
  if ( *(_BYTE *)(v14 + 16) <= 0x17u )
    goto LABEL_31;
  if ( *(_QWORD *)(v14 + 40) == *(_QWORD *)(v6 + 40) )
    goto LABEL_29;
  v11 = *(_QWORD *)(v14 + 8);
  if ( v11 )
  {
    if ( !*(_QWORD *)(v11 + 8) )
    {
LABEL_29:
      v69 = *(_QWORD *)(v6 - 24);
      v70 = a2[1];
      v11 = sub_1C5B330(a1, &v69, a3);
      if ( v11 )
      {
        v74 = 1;
        v71 = "newCast";
        v73 = 3;
        v11 = sub_12AA3B0(
                (__int64 *)v75,
                (unsigned int)*(unsigned __int8 *)(v6 + 16) - 24,
                v11,
                *(_QWORD *)v6,
                (__int64)&v71);
      }
      goto LABEL_31;
    }
    goto LABEL_33;
  }
LABEL_31:
  v15 = *a3;
LABEL_32:
  v16 = sub_1C5B220(v66, a2);
  v16[2] = v11;
  v16[3] = v15;
LABEL_10:
  if ( v75[0] )
    sub_161E7C0((__int64)v75, (__int64)v75[0]);
  return v11;
}
