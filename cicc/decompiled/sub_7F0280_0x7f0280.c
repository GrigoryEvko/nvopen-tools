// Function: sub_7F0280
// Address: 0x7f0280
//
__int64 __fastcall sub_7F0280(__int64 a1, __int64 a2)
{
  __m128i *v3; // r12
  __m128i *v4; // r14
  unsigned __int8 v5; // di
  char v6; // dl
  _QWORD *v8; // rax
  __int64 v9; // rax
  _BYTE *v10; // rax
  unsigned __int8 v11; // di
  _BYTE *v12; // r12
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  int v16; // r13d
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  _BYTE *v26; // rax
  unsigned int v27; // ecx
  __int64 v28; // r15
  _QWORD *v29; // rax
  __int64 v30; // rax
  _BYTE *v31; // r15
  void *v32; // rax
  unsigned int v33; // ecx
  __int64 v34; // r15
  _BYTE *v35; // r12
  __int64 *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  _BYTE *v41; // rax
  int v42; // ecx
  __int64 v43; // r15
  _QWORD *v44; // rax
  __int64 v45; // rax
  _BYTE *v46; // r15
  _QWORD *v47; // r15
  __int64 v48; // rax
  __int64 *v49; // r15
  __int64 v50; // rax
  _BYTE *v51; // rax
  __int64 v52; // rsi
  __int64 *v53; // r15
  _QWORD *v54; // rax
  __int64 v55; // rsi
  _QWORD *v56; // r15
  void *v57; // rax
  _BYTE *v58; // r12
  _BOOL4 v59; // eax
  _QWORD *v60; // [rsp+8h] [rbp-68h]
  __int64 v61; // [rsp+10h] [rbp-60h]
  unsigned int v62; // [rsp+10h] [rbp-60h]
  _BYTE *v63; // [rsp+18h] [rbp-58h]
  const __m128i *v64; // [rsp+18h] [rbp-58h]
  int v65; // [rsp+20h] [rbp-50h]
  unsigned int v66; // [rsp+20h] [rbp-50h]
  unsigned int v67; // [rsp+20h] [rbp-50h]
  unsigned int v68; // [rsp+20h] [rbp-50h]
  _QWORD *v69; // [rsp+20h] [rbp-50h]
  unsigned int v70; // [rsp+20h] [rbp-50h]
  unsigned int v71; // [rsp+20h] [rbp-50h]
  _QWORD *v72; // [rsp+28h] [rbp-48h]
  __int64 v73; // [rsp+30h] [rbp-40h]
  char v74; // [rsp+3Fh] [rbp-31h]

  v3 = *(__m128i **)(a1 + 72);
  v73 = *(_QWORD *)a1;
  v4 = (__m128i *)v3[1].m128i_i64[0];
  v74 = *(_BYTE *)(a1 + 56);
  v65 = unk_4F0687C;
  if ( (unsigned int)sub_7E1F90(v3->m128i_i64[0]) )
  {
    v72 = sub_72BA30(5u);
    sub_7E1D00(5, a2);
    v15 = qword_4F189F0;
    if ( v3[1].m128i_i8[8] != 2 && (a2 & 1) == 0 )
      sub_7EE560(v3, 0);
    if ( v4[1].m128i_i8[8] == 2 )
    {
      if ( v3[1].m128i_i8[8] != 2 )
      {
        v3 = v4;
        v4 = *(__m128i **)(a1 + 72);
      }
    }
    else
    {
      sub_7EE560(v4, 0);
    }
    v63 = sub_7EBF80(v3, v15, 0, 0);
    *((_QWORD *)v63 + 2) = sub_7EBF80(v4, v15, 0, 0);
    v16 = (unsigned __int8)((v74 == 59) + 58);
    v61 = (__int64)v63;
    v64 = (const __m128i *)sub_73DBF0(v16, (__int64)v72, (__int64)v63);
    if ( v65 )
    {
      if ( (unsigned int)sub_731770((__int64)v3, 0, v17, v18, v19, v20) )
      {
        v41 = sub_7EBF80(v3, v15, 1, 1u);
        v42 = 1;
      }
      else
      {
        v71 = sub_731770((__int64)v4, 0, v37, v38, v39, v40) != 0;
        v41 = sub_7EBF80(v3, v15, 1, v71);
        v42 = v71;
      }
      v43 = (__int64)v41;
      v68 = v42;
      v44 = sub_72BA30(unk_4F06A60);
      v45 = sub_8D6540(v44);
      v46 = sub_73E110(v43, v45);
      *((_QWORD *)v46 + 2) = sub_7E0E90(0, unk_4F06A60);
      v60 = sub_73DBF0(v16, (__int64)v72, (__int64)v46);
      v47 = sub_7EBF80(v3, qword_4F189F8, 1, v68);
      v48 = sub_8D6540(*v47);
      v49 = (__int64 *)sub_73E130(v47, v48);
      v62 = v68;
      v69 = sub_7EBF80(v4, qword_4F189F8, 1, v68);
      v50 = sub_8D6540(*v69);
      v51 = sub_73E130(v69, v50);
      v52 = *v49;
      v49[2] = (__int64)v51;
      v53 = (__int64 *)sub_73DBF0(0x38u, v52, (__int64)v49);
      v54 = sub_7E0E90(1, unk_4F06A60);
      v55 = *v53;
      v53[2] = (__int64)v54;
      v56 = sub_73DBF0(0x37u, v55, (__int64)v53);
      v56[2] = sub_7E0E90(0, unk_4F06A60);
      v60[2] = sub_73DBF0(v16, (__int64)v72, (__int64)v56);
      v57 = sub_73DBF0((v74 == 59) + 87, (__int64)v72, (__int64)v60);
      v33 = v62;
      v34 = (__int64)v57;
    }
    else
    {
      v21 = v61;
      if ( *(_BYTE *)(v61 + 24) == 2 )
      {
        v59 = sub_70FCE0(*(_QWORD *)(v61 + 56));
        v21 = v61;
        if ( v59 )
        {
          if ( (unsigned int)sub_711520(*(_QWORD *)(v61 + 56), (__int64)v72, v17, v18, v61) )
          {
            sub_730620(a1, v64);
            goto LABEL_9;
          }
        }
      }
      if ( (unsigned int)sub_731770((__int64)v3, 0, v17, v18, v21, v20) )
      {
        v26 = sub_7EBF80(v3, v15, 1, 1u);
        v27 = 1;
      }
      else
      {
        v70 = sub_731770((__int64)v4, 0, v22, v23, v24, v25) != 0;
        v26 = sub_7EBF80(v3, v15, 1, v70);
        v27 = v70;
      }
      v28 = (__int64)v26;
      v66 = v27;
      if ( v26[24] == 2 )
      {
        v58 = sub_7EBF80(v3, qword_4F189F8, 1, v27);
        *((_QWORD *)v58 + 2) = sub_7EBF80(v4, qword_4F189F8, 1, v66);
        v36 = (__int64 *)sub_73DBF0(v16, (__int64)v72, (__int64)v58);
LABEL_24:
        v64[1].m128i_i64[0] = (__int64)v36;
        sub_73D8E0(a1, (v74 == 59) + 87, *v36, 0, (__int64)v64);
        goto LABEL_9;
      }
      v29 = sub_72BA30(unk_4F06A60);
      v30 = sub_8D6540(v29);
      v31 = sub_73E110(v28, v30);
      *((_QWORD *)v31 + 2) = sub_7E0E90(0, unk_4F06A60);
      v32 = sub_73DBF0(v16, (__int64)v72, (__int64)v31);
      v33 = v66;
      v34 = (__int64)v32;
    }
    v67 = v33;
    v35 = sub_7EBF80(v3, qword_4F189F8, 1, v33);
    *((_QWORD *)v35 + 2) = sub_7EBF80(v4, qword_4F189F8, 1, v67);
    v36 = (__int64 *)sub_73DBF0(v16, (__int64)v72, (__int64)v35);
    if ( v34 )
    {
      *(_QWORD *)(v34 + 16) = v36;
      v36 = (__int64 *)sub_73DBF0((v74 != 59) + 87, (__int64)v72, v34);
    }
    goto LABEL_24;
  }
  if ( !(_DWORD)a2 )
    sub_7EE560(v3, 0);
  sub_7EE560(v4, 0);
  v5 = byte_4D03F80[0];
  if ( byte_4D03F80[0] <= 4u )
  {
    v3[1].m128i_i64[0] = 0;
    v8 = sub_72BA30(v5);
    v9 = sub_8D6540(v8);
    v10 = sub_73E110((__int64)v3, v9);
    v11 = byte_4D03F80[0];
    *(_QWORD *)(a1 + 72) = v10;
    v12 = v10;
    v13 = sub_72BA30(v11);
    v14 = sub_8D6540(v13);
    *((_QWORD *)v12 + 2) = sub_73E110((__int64)v4, v14);
  }
  v6 = v74;
  if ( v74 != 59 )
    v6 = 58;
  *(_BYTE *)(a1 + 56) = v6;
LABEL_9:
  *(_QWORD *)a1 = v73;
  return v73;
}
