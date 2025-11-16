// Function: sub_7F1E10
// Address: 0x7f1e10
//
char __fastcall sub_7F1E10(__m128i *a1, __m128i *a2, __m128i *a3, _DWORD *a4)
{
  __m128i *v4; // r15
  __m128i *v5; // r12
  __int8 v6; // r13
  __m128i *v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // r14
  __int64 i; // rax
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // rax
  __m128i *v14; // r10
  __m128i *v15; // r13
  char v16; // al
  __m128i *v17; // rdx
  __int64 v18; // rax
  __m128i *v19; // rdx
  _QWORD *v20; // r15
  __m128i *v21; // r10
  __int64 j; // rax
  _QWORD *v23; // rdi
  __int64 v24; // rax
  _BYTE *v25; // r15
  __int64 v26; // rax
  __int64 v27; // r15
  unsigned int *v28; // rax
  __int64 v29; // rcx
  char v30; // si
  _QWORD *v31; // rax
  __int64 v32; // rax
  const __m128i *v33; // r14
  __int64 v34; // rax
  __int64 *v35; // r13
  __int64 v36; // rbx
  __int64 *v37; // r15
  char v38; // al
  __int64 *v39; // rbx
  __int64 v40; // r15
  __int64 *v41; // r13
  _QWORD *v42; // rax
  __int64 v43; // rdx
  void *v44; // rbx
  __int64 *v45; // rax
  const __m128i *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r14
  void *v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // rax
  __m128i *v52; // r13
  __int64 k; // rdi
  __int64 v54; // r14
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __m128i *v58; // r15
  const __m128i *v59; // rax
  __int64 v60; // rax
  const __m128i *v61; // rax
  __int64 v62; // rax
  _BYTE *v63; // rax
  unsigned __int8 v67; // [rsp+1Fh] [rbp-A1h]
  __m128i *v68; // [rsp+20h] [rbp-A0h]
  __m128i *v69; // [rsp+20h] [rbp-A0h]
  __m128i *v70; // [rsp+20h] [rbp-A0h]
  __int64 v71; // [rsp+20h] [rbp-A0h]
  __int64 v72; // [rsp+28h] [rbp-98h]
  __m128i *v73; // [rsp+28h] [rbp-98h]
  __m128i *v74; // [rsp+28h] [rbp-98h]
  __m128i *v75; // [rsp+28h] [rbp-98h]
  __m128i *v76; // [rsp+28h] [rbp-98h]
  __int64 v77; // [rsp+38h] [rbp-88h] BYREF
  __int64 v78; // [rsp+40h] [rbp-80h] BYREF
  __int64 v79; // [rsp+48h] [rbp-78h] BYREF
  int v80; // [rsp+50h] [rbp-70h] BYREF
  __int64 v81; // [rsp+58h] [rbp-68h]
  __int64 v82[10]; // [rsp+70h] [rbp-50h] BYREF

  v4 = a2;
  v5 = a1;
  v6 = a1[3].m128i_i8[8];
  if ( a4 )
    *a4 = 0;
  sub_7EC2F0((__int64)a1, (__int64)a2);
  sub_7EAF80(a1->m128i_i64[0], a2);
  v7 = (__m128i *)a1[4].m128i_i64[1];
  if ( qword_4F0688C )
  {
    if ( v7[1].m128i_i8[8] == 20 )
    {
      v47 = v7[3].m128i_i64[1];
      if ( (unsigned __int8)(*(_BYTE *)(v47 + 174) - 1) <= 1u )
      {
        a1->m128i_i64[0] = sub_7F8700(*(_QWORD *)(v47 + 152));
        v48 = sub_72CBE0();
        v49 = sub_730FF0(a1);
        a2 = (__m128i *)sub_73E110((__int64)v49, v48);
        sub_730620((__int64)a1, a2);
        v5 = (__m128i *)a1[4].m128i_i64[1];
        v7 = (__m128i *)v5[4].m128i_i64[1];
      }
    }
  }
  v8 = v7->m128i_i64[0];
  v67 = v6 - 108;
  if ( (unsigned __int8)(v6 - 108) <= 1u )
    v9 = sub_7E1F00(v8);
  else
    v9 = sub_8D46C0(v8);
  while ( *(_BYTE *)(v9 + 140) == 12 )
    v9 = *(_QWORD *)(v9 + 160);
  sub_7EAF80(v9, a2);
  for ( i = v9; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v11 = *(_QWORD *)(i + 168);
  v12 = *(_BYTE *)(v11 + 16);
  if ( (v12 & 0x40) == 0 && (v12 & 0x20) != 0 )
    *(_BYTE *)(v11 + 16) = v12 & 0x3F | 0x40;
  v72 = *(_QWORD *)(v9 + 168);
  sub_7EE560(v7, 0);
  v13 = v72;
  v14 = (__m128i *)v7[1].m128i_i64[0];
  v15 = *(__m128i **)(v72 + 40);
  if ( !v15 )
  {
    v17 = v7;
    if ( (*(_BYTE *)(v72 + 16) & 0x40) == 0 )
    {
      v76 = (__m128i *)v7[1].m128i_i64[0];
      v27 = sub_72B0F0((__int64)v7, 0);
      sub_7E1790((__int64)&v80);
      sub_7F1A60(v76, v9, v27, 0, dword_4D04810, (v5[3].m128i_i8[12] & 2) != 0, 0, &v80);
      goto LABEL_28;
    }
    goto LABEL_20;
  }
  v15 = (__m128i *)v7[1].m128i_i64[0];
  if ( (v5[3].m128i_i64[1] & 0x208000000LL) != 0x208000000LL )
  {
    v15 = 0;
    v71 = v72;
    v75 = (__m128i *)v7[1].m128i_i64[0];
    sub_7EE560(v75, (__m128i *)1);
    v13 = v71;
    v14 = v75;
  }
  v16 = *(_BYTE *)(v13 + 16);
  v17 = v7;
  if ( v16 < 0 )
    v17 = v14;
  v14 = (__m128i *)v14[1].m128i_i64[0];
  if ( (v16 & 0x40) != 0 )
  {
LABEL_20:
    if ( !v4 )
      goto LABEL_80;
    v68 = v14;
    v73 = v17;
    v18 = sub_7F98A0(v4, 0);
    v19 = v73;
    v20 = (_QWORD *)v18;
    v21 = v68;
    for ( j = v9; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    if ( (*(_BYTE *)(j - 8) & 8) != 0 && (v29 = *(_QWORD *)(j + 168), v30 = *(_BYTE *)(v29 + 16), (v30 & 0x40) != 0) )
    {
      v31 = *(_QWORD **)v29;
      if ( *(_QWORD *)(v29 + 40) && v30 < 0 )
        v31 = (_QWORD *)*v31;
      v32 = sub_8D46C0(v31[1]);
      v19 = v73;
      v21 = v68;
      v23 = (_QWORD *)v32;
    }
    else
    {
      v23 = *(_QWORD **)(j + 160);
    }
    v69 = v21;
    v74 = v19;
    v24 = sub_72D2E0(v23);
    v25 = sub_73E130(v20, v24);
    v26 = sub_72CBE0();
    v14 = v69;
    v5->m128i_i64[0] = v26;
    *((_QWORD *)v25 + 2) = v74[1].m128i_i64[0];
    v74[1].m128i_i64[0] = (__int64)v25;
  }
  v70 = v14;
  v27 = sub_72B0F0((__int64)v7, 0);
  sub_7E1790((__int64)&v80);
  sub_7F1A60(v70, v9, v27, 0, dword_4D04810, (v5[3].m128i_i8[12] & 2) != 0, (__int64)v15, &v80);
  if ( v15 )
  {
    sub_7EE560(v15, (__m128i *)1);
    v5[3].m128i_i8[12] &= ~2u;
  }
LABEL_28:
  if ( v27 && (unsigned __int8)(*(_BYTE *)(v27 + 174) - 1) <= 1u )
  {
    v51 = sub_7FDF40(v27, 1, 0);
    v7[3].m128i_i64[1] = v51;
    v7->m128i_i64[0] = sub_72D2E0(*(_QWORD **)(v51 + 152));
  }
  if ( (v5[3].m128i_i8[11] & 0x40) != 0 )
  {
    LOBYTE(v28) = sub_7EBD50(v5->m128i_i64);
    goto LABEL_37;
  }
  if ( v67 > 1u )
  {
    v5[3].m128i_i8[8] = 105;
    sub_825720(v5);
    v28 = &dword_4D04380;
    if ( dword_4D04380 )
    {
      if ( v80 != 5 )
      {
        if ( v5[1].m128i_i8[8] == 1 )
        {
LABEL_36:
          if ( v5[3].m128i_i8[8] != 105 )
            goto LABEL_37;
          v28 = (unsigned int *)sub_72B0F0(v5[4].m128i_i64[1], 0);
          if ( !v28 || *((_BYTE *)v28 + 174) || !*((_WORD *)v28 + 88) )
            goto LABEL_37;
          LOWORD(v28) = *(_WORD *)(sub_72B0F0(v5[4].m128i_i64[1], 0) + 176);
          if ( (_WORD)v28 )
          {
            if ( (_WORD)v28 == 24989 )
            {
              v52 = *(__m128i **)(v5[4].m128i_i64[1] + 16);
              for ( k = v52->m128i_i64[0]; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                ;
              v54 = sub_8D46C0(k);
              if ( (*(_BYTE *)(v54 + 140) & 0xFD) == 8 )
              {
                sub_7E1790((__int64)v82);
                if ( (unsigned int)sub_731920((__int64)v52, 0, 0, v55, v56, v57) )
                {
                  v62 = sub_72CBE0();
                  v63 = sub_73E110((__int64)v52, v62);
                  sub_7E25D0((__int64)v63, (int *)v82);
                }
                else
                {
                  v58 = sub_7E7ED0(v52);
                  sub_7E25D0((__int64)v52, (int *)v82);
                  v52 = (__m128i *)sub_73E830((__int64)v58);
                }
                v59 = (const __m128i *)sub_7E23D0(v52);
                sub_7E2820(v59, v54, 0, 0, (int *)v82);
                LOBYTE(v28) = sub_730620((__int64)v5, (const __m128i *)v82[1]);
              }
              else
              {
                v60 = sub_72CBE0();
                v61 = (const __m128i *)sub_73E110((__int64)v52, v60);
                LOBYTE(v28) = sub_730620((__int64)v5, v61);
              }
            }
LABEL_37:
            if ( v80 == 5 )
              return (char)v28;
            goto LABEL_52;
          }
LABEL_80:
          sub_721090();
        }
LABEL_52:
        v45 = (__int64 *)sub_730FF0(v5);
        v46 = (const __m128i *)sub_73DF90(v81, v45);
        LOBYTE(v28) = sub_730620((__int64)v5, v46);
        return (char)v28;
      }
      LOBYTE(v28) = sub_76EF80(v5, a3, a4);
    }
    if ( v5[1].m128i_i8[8] != 1 )
      goto LABEL_37;
    goto LABEL_36;
  }
  v33 = (const __m128i *)v5[4].m128i_i64[1];
  v78 = 0;
  v34 = sub_7E1F00(v33->m128i_i64[0]);
  v35 = (__int64 *)v33[1].m128i_i64[0];
  v36 = *(_QWORD *)(v34 + 168);
  v37 = (__int64 *)v35[2];
  if ( dword_4D04810 && (sub_7E6EE0(v33[1].m128i_i64[0], (__int64)v33) || sub_7E6EE0((__int64)v33, (__int64)v35)) )
    v35 = (__int64 *)sub_7FA3E0(v35, &v33[1], &v78);
  v38 = *(_BYTE *)(v36 + 16);
  if ( (v38 & 0x40) != 0 )
  {
    v39 = v37;
    if ( v38 >= 0 )
    {
      v39 = v35;
      v35 = v37;
    }
    v40 = v37[2];
    v39[2] = 0;
    v33[1].m128i_i64[0] = 0;
    v35[2] = 0;
    v41 = (__int64 *)sub_7E80B0(v35, v33, &v79, v82, &v77);
    v42 = sub_73E830(v79);
    v43 = v77;
    *(_QWORD *)(v77 + 16) = v39;
    v39[2] = (__int64)v42;
    v42[2] = v40;
  }
  else
  {
    v33[1].m128i_i64[0] = 0;
    v35[2] = 0;
    v41 = (__int64 *)sub_7E80B0(v35, v33, &v79, v82, &v77);
    v50 = sub_73E830(v79);
    v43 = v77;
    *(_QWORD *)(v77 + 16) = v50;
    v50[2] = v37;
  }
  v44 = sub_73DBF0(0x69u, v5->m128i_i64[0], v43);
  if ( v78 )
    v41 = (__int64 *)sub_73DF90(v78, v41);
  v41[2] = (__int64)v44;
  LOBYTE(v28) = sub_73D8E0((__int64)v5, 0x5Bu, v5->m128i_i64[0], v5[1].m128i_i8[9] & 1, (__int64)v41);
  if ( v80 != 5 )
    goto LABEL_52;
  return (char)v28;
}
