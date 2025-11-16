// Function: sub_808590
// Address: 0x808590
//
__int16 __fastcall sub_808590(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // r15
  char *v3; // rax
  char *v4; // rdi
  __int64 v5; // r12
  _QWORD *v6; // rbx
  __int64 *v7; // r14
  int v8; // r12d
  __int64 *v9; // r15
  _QWORD *v10; // rax
  __int64 v11; // r14
  _QWORD *v12; // rax
  __m128i *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  size_t v17; // rax
  char *v18; // rax
  int v19; // r14d
  __int64 v20; // rdi
  __int64 *v21; // rbx
  __int64 v22; // rax
  __m128i *v23; // r12
  __int64 v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rbx
  unsigned __int64 v30; // r12
  __int64 v31; // rdi
  unsigned __int64 v32; // rcx
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rcx
  __int64 v35; // rdi
  __m128i *v36; // r14
  _BYTE *v37; // r15
  _QWORD *v38; // rax
  __int64 v39; // rax
  char i; // dl
  _QWORD *v41; // rax
  int v42; // r14d
  __int64 j; // rax
  __int64 v44; // r12
  _QWORD *v45; // rax
  __m128i *v46; // r12
  _BYTE *v47; // rbx
  _QWORD *v48; // rax
  _QWORD *v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdx
  int *v53; // rax
  int *v54; // rax
  __int16 v55; // cx
  __int8 v56; // al
  __int64 v57; // rax
  size_t v58; // rax
  char *v59; // rbx
  bool v60; // zf
  _QWORD *v61; // rax
  __m128i *v62; // r15
  __m128i *v63; // r14
  __int64 v65; // [rsp-1A0h] [rbp-1A0h]
  const char *v66; // [rsp-190h] [rbp-190h]
  __int64 *v67; // [rsp-188h] [rbp-188h]
  __m128i *v68; // [rsp-178h] [rbp-178h]
  _QWORD *m128i_i64; // [rsp-170h] [rbp-170h]
  __int64 v70; // [rsp-170h] [rbp-170h]
  unsigned int v71; // [rsp-15Ch] [rbp-15Ch] BYREF
  int v72[8]; // [rsp-158h] [rbp-158h] BYREF
  char v73[64]; // [rsp-138h] [rbp-138h] BYREF
  _BYTE v74[248]; // [rsp-F8h] [rbp-F8h] BYREF

  v1 = a1[24] & 0x24000000002000LL;
  if ( v1 == 0x20000000002000LL )
  {
    v2 = (__int64)a1;
    v3 = (char *)a1[2];
    v4 = (char *)a1[1];
    if ( !v3 )
      v3 = v4;
    v66 = v3;
    if ( (*(_BYTE *)(v2 + 197) & 0x60) == 0x20 )
    {
      v4 = *(char **)(v2 + 24);
      *(_BYTE *)(v2 + 89) &= ~8u;
      *(_QWORD *)(v2 + 24) = 0;
      *(_QWORD *)(v2 + 8) = v4;
    }
    v5 = *(_QWORD *)(v2 + 152);
    v68 = sub_7F7840(v4, *(_BYTE *)(v2 + 172), *(_QWORD *)(v5 + 160), 0);
    sub_7362F0((__int64)v68, 0);
    v67 = sub_7F54F0((__int64)v68, 0, 0, &v71);
    sub_7E1740(v67[10], (__int64)v72);
    sub_7F6C60((__int64)v67, v71, (__int64)v74);
    m128i_i64 = v67 + 5;
    v6 = *(_QWORD **)(v68[9].m128i_i64[1] + 168);
    v7 = **(__int64 ***)(v5 + 168);
    if ( v7 )
    {
      v65 = v2;
      v8 = 0;
      v9 = v7;
      do
      {
        if ( (unsigned int)sub_8D2FB0(v9[1]) )
        {
          v10 = (_QWORD *)sub_8D46C0(v9[1]);
          v11 = sub_72D2E0(v10);
        }
        else
        {
          v11 = (__int64)sub_73C570((const __m128i *)v9[1], (*((_DWORD *)v9 + 8) >> 11) & 0x7F);
        }
        v12 = sub_724EF0(v11);
        *v6 = v12;
        v6 = v12;
        *((_BYTE *)v12 + 32) = v9[4] & 2 | v12[4] & 0xFD;
        v13 = sub_7E2270(v11);
        *m128i_i64 = v13;
        m128i_i64 = v13[7].m128i_i64;
        v14 = sub_8E36B0(v13[7].m128i_i64[1]);
        *(_BYTE *)(v14 + 141) |= 2u;
        v15 = v9[2];
        if ( v15 )
        {
          v16 = sub_8E36B0(v15);
          *(_BYTE *)(v16 + 141) |= 2u;
        }
        sprintf(v73, "__par%d", v8++);
        v17 = strlen(v73);
        v18 = (char *)sub_7E1510(v17 + 1);
        v13->m128i_i64[1] = (__int64)strcpy(v18, v73);
        v9 = (__int64 *)*v9;
      }
      while ( v9 );
      v2 = v65;
      v19 = v8;
      v20 = v8;
      v21 = *(__int64 **)(v65 + 240);
      if ( !v21 )
        goto LABEL_21;
      do
      {
LABEL_17:
        while ( *((_BYTE *)v21 + 8) )
        {
          v21 = (__int64 *)*v21;
          if ( !v21 )
            goto LABEL_19;
        }
        v22 = sub_8E36B0(v21[4]);
        *(_BYTE *)(v22 + 141) |= 2u;
        v21 = (__int64 *)*v21;
      }
      while ( v21 );
LABEL_19:
      v20 = v19;
      if ( v19 )
      {
LABEL_21:
        v23 = (__m128i *)sub_73A830(v20, 5u);
        v24 = sub_72CBE0();
        v25 = sub_7F8900("__cudaLaunchPrologue", (__m128i **)&qword_4F18AA0, v24, v23);
        sub_7E69E0(v25, v72);
        v29 = v67[5];
        if ( v29 )
        {
          v70 = v2;
          v30 = 0;
          do
          {
            v31 = *(_QWORD *)(v29 + 120);
            if ( *(char *)(v31 + 142) >= 0 && *(_BYTE *)(v31 + 140) == 12 )
              v32 = (unsigned int)sub_8D4AB0(v31, v72, v26);
            else
              v32 = *(unsigned int *)(v31 + 136);
            v33 = v30 % v32;
            v34 = v30 + v32 - v30 % v32;
            if ( v33 )
              v30 = v34;
            if ( unk_4F068E0
              && (qword_4F077A8 > 0x9EFBu || qword_4F068D8 > 0x9EFBu)
              && ((unsigned int)sub_8D2E30(*(_QWORD *)(v29 + 120)) || (unsigned int)sub_8D2FB0(*(_QWORD *)(v29 + 120)))
              && (v35 = *(_QWORD *)(v29 + 120), (*(_BYTE *)(v35 + 140) & 0xFB) == 8)
              && (sub_8D4C10(v35, dword_4F077C4 != 2) & 4) != 0 )
            {
              v62 = (__m128i *)sub_73E830(v29);
              v63 = sub_7E7ED0(v62);
              sub_7604D0((__int64)v63, 7u);
              v63[5].m128i_i8[8] |= 4u;
              sub_7E69E0(v62, v72);
              v36 = (__m128i *)sub_73E830((__int64)v63);
            }
            else
            {
              v36 = (__m128i *)sub_73E830(v29);
            }
            v37 = sub_724D50(1);
            *((_QWORD *)v37 + 16) = sub_72BA30(byte_4F06A51[0]);
            sub_620DE0((_WORD *)v37 + 88, v30);
            v38 = sub_726700(2);
            v38[7] = v37;
            *v38 = *((_QWORD *)v37 + 16);
            v36[1].m128i_i64[0] = (__int64)v38;
            v39 = *(_QWORD *)(v29 + 120);
            for ( i = *(_BYTE *)(v39 + 140); i == 12; i = *(_BYTE *)(v39 + 140) )
              v39 = *(_QWORD *)(v39 + 160);
            if ( (unsigned __int8)(i - 9) <= 1u )
            {
              v61 = sub_72BA30(5u);
              v25 = sub_7F8900("__cudaSetupArg", (__m128i **)&qword_4F18A90, (__int64)v61, v36);
            }
            else
            {
              v41 = sub_72BA30(5u);
              v25 = sub_7F8900("__cudaSetupArgSimple", (__m128i **)&qword_4F18A88, (__int64)v41, v36);
            }
            v42 = unk_4F189C4;
            unk_4F189C4 = 1;
            sub_7E69E0(v25, v72);
            unk_4F189C4 = v42;
            for ( j = *(_QWORD *)(v29 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            v29 = *(_QWORD *)(v29 + 112);
            v30 += *(_QWORD *)(j + 128);
          }
          while ( v29 );
          v2 = v70;
        }
        v44 = sub_7E1C30(v25, v72, v26, v27, v28);
        v45 = sub_731330(v2);
        v46 = (__m128i *)sub_73E110((__int64)v45, v44);
        v47 = sub_724D50(1);
        *((_QWORD *)v47 + 16) = sub_72BA30(6u);
        sub_620DE0((_WORD *)v47 + 88, (*(_BYTE *)(v2 + 198) & 0x40) != 0);
        v48 = sub_726700(2);
        v48[7] = v47;
        *v48 = *((_QWORD *)v47 + 16);
        v46[1].m128i_i64[0] = (__int64)v48;
        v49 = sub_72BA30(5u);
        v50 = sub_7F8900("__cudaLaunch", (__m128i **)&qword_4F18A80, (__int64)v49, v46);
        sub_7E69E0(v50, v72);
        sub_7FB010((__int64)v67, v71, (__int64)v74);
        sub_7604D0(v2, 0xBu);
        sub_7605A0(v2);
        v51 = *(int *)(v2 + 160);
        *(_DWORD *)(v2 + 160) = v68[10].m128i_i32[0];
        v68[10].m128i_i32[0] = v51;
        v52 = unk_4F073B0;
        v53 = (int *)(unk_4F072B8 + 16 * v51);
        if ( !*(_QWORD *)(unk_4F073B0 + 8LL * v53[2]) )
        {
          MEMORY[0x20] = v68;
          BUG();
        }
        *(_QWORD *)(*(_QWORD *)v53 + 32LL) = v68;
        v54 = (int *)(unk_4F072B8 + 16LL * *(int *)(v2 + 160));
        if ( !*(_QWORD *)(v52 + 8LL * v54[2]) )
        {
          MEMORY[0x20] = v2;
          BUG();
        }
        *(_QWORD *)(*(_QWORD *)v54 + 32LL) = v2;
        v68[4].m128i_i64[0] = *(_QWORD *)(v2 + 64);
        *(_QWORD *)(v2 + 64) = *(_QWORD *)&dword_4F077C8;
        v55 = v68[12].m128i_i16[3];
        v68[20].m128i_i64[1] = *(_QWORD *)(v2 + 328);
        v68[12].m128i_i16[3] = v55 & 0xDFDF | (((*(_BYTE *)(v2 + 199) & 0x20) != 0) << 13) | 0x20;
        if ( (*(_BYTE *)(v2 + 198) & 0x40) != 0 )
        {
          v56 = v68[12].m128i_i8[6] | 0x40;
          v68[12].m128i_i8[6] = v56;
        }
        else
        {
          v56 = v68[12].m128i_i8[6];
        }
        v68[12].m128i_i8[6] = v56 | 0x10;
        v68[12].m128i_i8[7] = *(_BYTE *)(v2 + 199) & 8 | v68[12].m128i_i8[7] & 0xF7;
        *(_BYTE *)(v2 + 199) &= ~8u;
        v57 = qword_4F18A30;
        v68[8].m128i_i64[0] = v2;
        v68[12].m128i_i8[1] &= ~0x10u;
        v68[7].m128i_i64[1] = v57;
        qword_4F18A30 = (__int64)v68;
        *(_BYTE *)(v2 + 193) |= 0x10u;
        *(_QWORD *)(v2 + 128) = v68;
        v58 = strlen(v66);
        v59 = (char *)sub_7E1510(v58 + 15);
        sprintf(v59, "__device_stub_%s", v66);
        LODWORD(v1) = *(_DWORD *)(v2 + 196);
        *(_QWORD *)(v2 + 8) = v59;
        *(_QWORD *)(v2 + 328) = 0;
        LODWORD(v1) = v1 & 0xDF8F9FFF;
        BYTE1(v1) |= 0x20u;
        v60 = *(_QWORD *)(v2 + 248) == 0;
        *(_DWORD *)(v2 + 196) = v1;
        if ( !v60 )
        {
          *(_BYTE *)(v2 + 200) &= 0xF8u;
          *(_BYTE *)(v2 + 172) = 2;
          goto LABEL_48;
        }
        v1 = (__int64)&dword_4F077C4;
        if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || (v1 = (__int64)&dword_4F07774, dword_4F07774)) )
        {
          LODWORD(v1) = sub_736990(v2);
          if ( (_DWORD)v1 )
          {
            *(_BYTE *)(v2 + 172) = 0;
          }
          else if ( *(_BYTE *)(v2 + 172) == 2 )
          {
            goto LABEL_48;
          }
          if ( *(_QWORD *)(v2 + 248) )
          {
LABEL_48:
            *(_BYTE *)(v2 + 205) &= ~2u;
            return v1;
          }
        }
        else if ( *(_BYTE *)(v2 + 172) == 2 )
        {
          goto LABEL_48;
        }
        v1 = (__int64)&dword_4F068C4;
        if ( dword_4F068C4 )
        {
          if ( (unsigned int)sub_736990(v2) && !unk_4D04724 )
            *(_BYTE *)(v2 + 200) = *(_BYTE *)(v2 + 200) & 0xF8 | 1;
          LOWORD(v1) = dword_4D04714;
          if ( dword_4D04714 )
          {
            LOWORD(v1) = *(unsigned __int8 *)(v2 + 200);
            if ( (v1 & 7) == 0 )
            {
              LOWORD(v1) = v1 & 0xFFF8 | 1;
              *(_BYTE *)(v2 + 200) = v1;
            }
          }
        }
        goto LABEL_48;
      }
    }
    else
    {
      v21 = *(__int64 **)(v2 + 240);
      if ( v21 )
      {
        v19 = 0;
        goto LABEL_17;
      }
    }
    v20 = 1;
    goto LABEL_21;
  }
  return v1;
}
