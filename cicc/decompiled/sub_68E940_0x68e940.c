// Function: sub_68E940
// Address: 0x68e940
//
__int64 __fastcall sub_68E940(const __m128i *a1, __int64 **a2, _DWORD *a3, __int64 a4, _QWORD *a5)
{
  __int64 **v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r13
  int v14; // eax
  __int64 *v15; // rdi
  _DWORD *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 *v24; // r15
  __int64 i; // r15
  char v26; // dl
  __int64 v27; // r11
  __int64 v28; // rax
  int v29; // ecx
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // edi
  __int8 v34; // al
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 k; // rdx
  __int64 v38; // rax
  __int64 *v39; // r8
  __int64 v40; // rsi
  _BOOL4 v41; // eax
  __int64 v42; // rdx
  __int64 v43; // rcx
  _QWORD **v44; // r8
  _QWORD **v45; // rdi
  _DWORD *v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // r8
  _BOOL4 v52; // eax
  __int64 v53; // rax
  int v54; // eax
  int v55; // eax
  __int64 v56; // rdx
  __int64 j; // rax
  unsigned __int16 v58; // ax
  char v59; // al
  __int64 v60; // rax
  char v61; // dl
  int v62; // eax
  __int64 v63; // rdx
  __int64 v64; // rax
  int v65; // eax
  int v66; // eax
  __int64 v67; // rdx
  int v68; // eax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 **v71; // rdi
  _DWORD *v72; // r12
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  int v77; // eax
  __int64 v78; // [rsp-8h] [rbp-1B8h]
  int v79; // [rsp+8h] [rbp-1A8h]
  int v80; // [rsp+8h] [rbp-1A8h]
  int v81; // [rsp+8h] [rbp-1A8h]
  __int64 *v82; // [rsp+10h] [rbp-1A0h]
  __int64 v83; // [rsp+10h] [rbp-1A0h]
  int v84; // [rsp+10h] [rbp-1A0h]
  __int64 v85; // [rsp+10h] [rbp-1A0h]
  __int64 v86; // [rsp+10h] [rbp-1A0h]
  __int64 v87; // [rsp+10h] [rbp-1A0h]
  __int64 v89; // [rsp+18h] [rbp-198h]
  _OWORD v90[5]; // [rsp+60h] [rbp-150h] BYREF
  __m128i v91; // [rsp+B0h] [rbp-100h]
  __m128i v92; // [rsp+C0h] [rbp-F0h]
  __m128i v93; // [rsp+D0h] [rbp-E0h]
  __m128i v94; // [rsp+E0h] [rbp-D0h]
  __m128i v95; // [rsp+F0h] [rbp-C0h]
  __m128i v96; // [rsp+100h] [rbp-B0h]
  __m128i v97; // [rsp+110h] [rbp-A0h]
  __m128i v98; // [rsp+120h] [rbp-90h]
  __m128i v99; // [rsp+130h] [rbp-80h]
  __m128i v100; // [rsp+140h] [rbp-70h]
  __m128i v101; // [rsp+150h] [rbp-60h]
  __m128i v102; // [rsp+160h] [rbp-50h]
  __m128i v103; // [rsp+170h] [rbp-40h]

  sub_72C930(a1);
  v13 = sub_6EB5C0(a1);
  *a5 = 0;
  if ( !a2 )
    goto LABEL_35;
  v14 = *(_DWORD *)(a4 + 32);
  if ( v14 == 1 )
  {
    v15 = *a2;
    if ( *a2 )
    {
LABEL_7:
      v16 = (_DWORD *)sub_6E1A20(v15);
      if ( (unsigned int)sub_6E5430(v15, a2, v17, v18, v19, v20) )
        sub_6851C0(0x8Cu, v16);
      return v13;
    }
    goto LABEL_13;
  }
  if ( v14 > 1 )
  {
    v9 = (__int64 **)*a2;
    if ( !*a2 )
      goto LABEL_35;
    if ( v14 == 2 )
      goto LABEL_6;
  }
  if ( v14 <= 2 )
    goto LABEL_13;
  v9 = (__int64 **)**a2;
  if ( !v9 )
  {
LABEL_35:
    if ( (unsigned int)sub_6E5430(a1, a2, v9, v10, v11, v12) )
      sub_6851C0(0xA5u, a3);
    return v13;
  }
  if ( v14 == 3 )
  {
LABEL_6:
    v15 = *v9;
    if ( *v9 )
      goto LABEL_7;
  }
LABEL_13:
  sub_6E65B0(a2);
  v24 = a2[3];
  if ( *((_BYTE *)v24 + 25) == 1 )
    sub_6FA3A0(v24 + 1);
  for ( i = v24[1]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( !*(_DWORD *)(a4 + 60) )
  {
    if ( (unsigned int)sub_8D2B80(i) )
    {
      v26 = *(_BYTE *)(i + 140);
      v53 = i;
      if ( v26 == 12 )
      {
        do
          v53 = *(_QWORD *)(v53 + 160);
        while ( *(_BYTE *)(v53 + 140) == 12 );
      }
      v27 = *(_QWORD *)(v53 + 160);
LABEL_20:
      if ( v26 != 12 )
        goto LABEL_23;
      goto LABEL_21;
    }
    v59 = *(_BYTE *)(i + 140);
    if ( v59 == 12 )
    {
      v60 = i;
      do
      {
        v60 = *(_QWORD *)(v60 + 160);
        v61 = *(_BYTE *)(v60 + 140);
      }
      while ( v61 == 12 );
      if ( !v61 )
      {
        LODWORD(v27) = i;
LABEL_21:
        v28 = i;
        do
        {
          v28 = *(_QWORD *)(v28 + 160);
          v26 = *(_BYTE *)(v28 + 140);
        }
        while ( v26 == 12 );
LABEL_23:
        if ( v26 )
        {
          v84 = v27;
          v54 = sub_8DBE70(i);
          LODWORD(v27) = v84;
          if ( !v54 )
          {
            v55 = sub_8D2B80(i);
            v56 = i;
            if ( v55 )
            {
              for ( j = i; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                ;
              v56 = *(_QWORD *)(j + 160);
            }
            v58 = *(_WORD *)(v13 + 176);
            if ( v58 <= 0x1225u )
            {
              if ( v58 > 0x1200u )
              {
                switch ( v58 )
                {
                  case 0x1202u:
                  case 0x1204u:
                  case 0x1205u:
                  case 0x1206u:
                  case 0x1208u:
                  case 0x1209u:
                  case 0x120Au:
                  case 0x120Bu:
                  case 0x120Cu:
                  case 0x120Du:
                  case 0x120Eu:
                  case 0x120Fu:
                  case 0x1210u:
                  case 0x1211u:
                  case 0x1212u:
                  case 0x1213u:
                  case 0x1214u:
                  case 0x1216u:
                  case 0x1218u:
                  case 0x1219u:
                  case 0x121Bu:
                  case 0x121Cu:
                  case 0x121Du:
                  case 0x121Eu:
                  case 0x121Fu:
                  case 0x1220u:
                  case 0x1221u:
                  case 0x1223u:
                  case 0x1224u:
                  case 0x1225u:
                    goto LABEL_68;
                  case 0x1203u:
                  case 0x1207u:
                  case 0x121Au:
                  case 0x1222u:
                    goto LABEL_72;
                  case 0x1215u:
                  case 0x1217u:
                    goto LABEL_74;
                  default:
                    v81 = v84;
                    v87 = v56;
                    v77 = sub_8D27E0(v56);
                    v67 = v87;
                    LODWORD(v27) = v81;
                    if ( !v77 )
                      goto LABEL_75;
                    goto LABEL_24;
                }
              }
LABEL_88:
              sub_721090(i);
            }
            switch ( v58 )
            {
              case 0x3D1Bu:
              case 0x3D1Cu:
              case 0x3D21u:
              case 0x3D22u:
              case 0x3D23u:
LABEL_72:
                v80 = v84;
                v86 = v56;
                v65 = sub_8D2780(v56);
                v63 = v86;
                LODWORD(v27) = v80;
                if ( !v65 )
                  goto LABEL_69;
                goto LABEL_24;
              case 0x3D1Du:
              case 0x3D1Fu:
LABEL_74:
                v81 = v84;
                v87 = v56;
                v66 = sub_8D2780(v56);
                v67 = v87;
                LODWORD(v27) = v81;
                if ( v66 )
                  goto LABEL_24;
LABEL_75:
                v68 = sub_8D2A90(v67);
                v63 = v87;
                LODWORD(v27) = v81;
                if ( v68 )
                  goto LABEL_24;
LABEL_69:
                v89 = v63;
                v64 = sub_6E1A20(a2);
                sub_6E5E80(3257, v64, v89);
                break;
              case 0x3D1Eu:
              case 0x3D20u:
LABEL_68:
                v79 = v84;
                v85 = v56;
                v62 = sub_8D2A90(v56);
                v63 = v85;
                LODWORD(v27) = v79;
                if ( !v62 )
                  goto LABEL_69;
                goto LABEL_24;
              default:
                goto LABEL_88;
            }
            return v13;
          }
        }
LABEL_24:
        if ( *(int *)(a4 + 32) <= 2 )
        {
          v29 = 0;
          v30 = 0;
          if ( *(_DWORD *)(a4 + 32) == 2 )
            v30 = i;
        }
        else
        {
          v29 = i;
          v30 = i;
        }
        v31 = sub_732700(v27, i, v30, v29, 0, 0, 0, 0);
        v32 = sub_68A000(v13, v31);
        v13 = *(_QWORD *)(v32 + 88);
        v33 = v32;
        v34 = a1[1].m128i_i8[0];
        v90[0] = _mm_loadu_si128(a1 + 4);
        v90[1] = _mm_loadu_si128(a1 + 5);
        v90[2] = _mm_loadu_si128(a1 + 6);
        v90[3] = _mm_loadu_si128(a1 + 7);
        v90[4] = _mm_loadu_si128(a1 + 8);
        if ( v34 == 2 )
        {
          v91 = _mm_loadu_si128(a1 + 9);
          v92 = _mm_loadu_si128(a1 + 10);
          v93 = _mm_loadu_si128(a1 + 11);
          v94 = _mm_loadu_si128(a1 + 12);
          v95 = _mm_loadu_si128(a1 + 13);
          v96 = _mm_loadu_si128(a1 + 14);
          v97 = _mm_loadu_si128(a1 + 15);
          v98 = _mm_loadu_si128(a1 + 16);
          v99 = _mm_loadu_si128(a1 + 17);
          v100 = _mm_loadu_si128(a1 + 18);
          v101 = _mm_loadu_si128(a1 + 19);
          v102 = _mm_loadu_si128(a1 + 20);
          v103 = _mm_loadu_si128(a1 + 21);
        }
        else if ( v34 == 5 || v34 == 1 )
        {
          v91.m128i_i64[0] = a1[9].m128i_i64[0];
        }
        sub_6EAB60(
          v33,
          (a1[1].m128i_i8[2] & 0x40) != 0,
          0,
          (unsigned int)v90 + 4,
          (unsigned int)v90 + 12,
          a1[5].m128i_i64[1],
          (__int64)a1);
        k = v78;
        if ( a1[1].m128i_i8[0] )
        {
          v38 = a1->m128i_i64[0];
          for ( k = *(unsigned __int8 *)(a1->m128i_i64[0] + 140); (_BYTE)k == 12; k = *(unsigned __int8 *)(v38 + 140) )
            v38 = *(_QWORD *)(v38 + 160);
          if ( (_BYTE)k )
            sub_6F5FA0(a1, 0, 0, 1, v35, v36);
        }
        *a5 = sub_6F6D20(a2, 0, k);
        return v13;
      }
    }
    else if ( !v59 )
    {
      LODWORD(v27) = i;
      goto LABEL_24;
    }
    if ( dword_4F04C44 == -1
      && (v69 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v69 + 6) & 6) == 0)
      && *(_BYTE *)(v69 + 4) != 12
      || !(unsigned int)sub_8DBE70(i) )
    {
      v70 = sub_6E1A20(a2);
      sub_6E5E80(2793, v70, i);
      return v13;
    }
LABEL_19:
    v26 = *(_BYTE *)(i + 140);
    LODWORD(v27) = i;
    goto LABEL_20;
  }
  if ( *(int *)(a4 + 32) <= 1 )
    goto LABEL_19;
  v39 = *a2;
  v40 = (*a2)[3];
  v82 = *a2;
  v41 = sub_68B7F0(i, v40, v22, v23, (__int64)v39);
  v44 = (_QWORD **)v82;
  if ( v41 )
  {
    if ( *(int *)(a4 + 32) <= 2 )
      goto LABEL_19;
    v51 = **a2;
    v40 = *(_QWORD *)(v51 + 24);
    v83 = v51;
    v52 = sub_68B7F0(i, v40, v42, v43, v51);
    v44 = (_QWORD **)v83;
    if ( v52 )
      goto LABEL_19;
    goto LABEL_41;
  }
  if ( *(_DWORD *)(a4 + 32) != 2 )
  {
LABEL_41:
    v45 = v44;
    v46 = (_DWORD *)sub_6E1A20(v44);
    if ( (unsigned int)sub_6E5430(v45, v40, v47, v48, v49, v50) )
      sub_6851C0(0xD7Cu, v46);
    return v13;
  }
  v71 = a2;
  v72 = (_DWORD *)sub_6E1A20(a2);
  if ( (unsigned int)sub_6E5430(v71, v40, v73, v74, v75, v76) )
    sub_6851C0(0xCB8u, v72);
  return v13;
}
