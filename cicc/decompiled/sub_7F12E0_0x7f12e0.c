// Function: sub_7F12E0
// Address: 0x7f12e0
//
__int64 __fastcall sub_7F12E0(__int64 a1)
{
  __int64 v2; // r14
  _QWORD *v3; // r15
  __m128i *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // rbx
  __m128i *v20; // rbx
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __m128i *v26; // r13
  __int64 v27; // rax
  unsigned __int8 v28; // di
  __m128i *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r13
  _QWORD *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rdi
  __m128i *v41; // rsi
  const __m128i *v42; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // [rsp+0h] [rbp-60h]
  _BYTE *v47; // [rsp+8h] [rbp-58h]
  int v48; // [rsp+10h] [rbp-50h]
  char v49; // [rsp+15h] [rbp-4Bh]
  unsigned __int8 v50; // [rsp+16h] [rbp-4Ah]
  char v51; // [rsp+17h] [rbp-49h]
  __int64 v52; // [rsp+18h] [rbp-48h]
  unsigned int v53; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v54[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 72);
  v3 = *(_QWORD **)(v2 + 16);
  v51 = *(_BYTE *)(a1 + 56);
  v4 = sub_73D720(*(const __m128i **)v2);
  v54[0] = 0;
  v46 = (__int64)v4;
  v48 = *(_BYTE *)(a1 + 58) & 1;
  v49 = *(_BYTE *)(a1 + 58) & 1;
  v9 = sub_73D850(a1);
  if ( dword_4D04810 && (*(_BYTE *)(a1 + 60) & 2) != 0 && (sub_7E6EE0(v2, (__int64)v3) || sub_7E6EE0((__int64)v3, v2)) )
    v3 = (_QWORD *)sub_7FA3E0(v3, v2 + 16, v54);
  *(_QWORD *)(v2 + 16) = 0;
  v10 = (unsigned int)sub_731770((__int64)v3, 0, v5, v6, v7, v8);
  v14 = sub_7E2590(v2, v10, &v53, v11, v12, v13);
  v18 = v53;
  v47 = (_BYTE *)v2;
  v52 = 0;
  v19 = v14;
  if ( v53 )
  {
    v52 = sub_7E7CB0(v46);
    v47 = sub_731250(v52);
    v45 = v2;
    v2 = v19;
    v19 = v45;
  }
  v20 = (__m128i *)sub_731370(v19, v10, v15, v18, v16, v17);
  v50 = v51 - 74;
  if ( !(unsigned int)sub_8D29A0(v20->m128i_i64[0]) || !(unsigned int)sub_8D2E30(*v3) )
  {
    if ( (unsigned int)sub_8D2B20(v9) )
    {
      if ( (unsigned __int8)(v51 - 76) <= 1u )
      {
        if ( (unsigned int)sub_8D2B20(v20->m128i_i64[0]) )
        {
          while ( *(_BYTE *)(v9 + 140) == 12 )
            v9 = *(_QWORD *)(v9 + 160);
          v9 = (__int64)sub_72C610(*(_BYTE *)(v9 + 160));
          v28 = a5678923destroy[v50];
LABEL_15:
          v20[1].m128i_i64[0] = (__int64)v3;
          v21 = v9;
          v26 = (__m128i *)sub_73DBF0(v28, v9, (__int64)v20);
          if ( dword_4F077C4 != 2 )
            goto LABEL_9;
          goto LABEL_16;
        }
        v20 = sub_7F1240(v20, v9);
        goto LABEL_13;
      }
      v20 = sub_7F1240(v20, v9);
      if ( v50 <= 0xBu )
      {
LABEL_13:
        v27 = v50;
LABEL_14:
        v28 = a5678923destroy[v27];
        goto LABEL_15;
      }
    }
    else
    {
      v20 = sub_7F1240(v20, v9);
      v27 = v50;
      if ( v50 <= 0xBu )
        goto LABEL_14;
    }
LABEL_32:
    sub_721090();
  }
  if ( v50 > 0xBu )
    goto LABEL_32;
  v3[2] = v20;
  v21 = v9;
  v26 = (__m128i *)sub_73DBF0(a5678923destroy[v50], v9, (__int64)v3);
  if ( dword_4F077C4 != 2 )
  {
LABEL_9:
    sub_7D9310(v26->m128i_i64, v21, v22, v23, v24, v25);
    goto LABEL_17;
  }
LABEL_16:
  if ( *(_BYTE *)(a1 + 57) == 5 )
    goto LABEL_9;
LABEL_17:
  v29 = sub_7F1240(v26, v46);
  v33 = sub_698020(v47, 73, (__int64)v29, v30, v31, v32);
  if ( v52 )
  {
    v34 = sub_73E830(v52);
    v38 = sub_698020((_QWORD *)v2, 73, (__int64)v34, v35, v36, v37);
    if ( v48 )
    {
      v39 = *(_QWORD *)a1;
      *(_BYTE *)(v38 + 25) |= 1u;
      *(_BYTE *)(v38 + 58) |= 1u;
      *(_QWORD *)v38 = v39;
    }
    v33 = (__int64)sub_73DF90(v33, (__int64 *)v38);
    *(_BYTE *)(v33 + 25) = v49 | *(_BYTE *)(v33 + 25) & 0xFE;
    *(_BYTE *)(v33 + 58) = v49 | *(_BYTE *)(v33 + 58) & 0xFE;
  }
  else if ( v48 )
  {
    v44 = *(_QWORD *)a1;
    *(_BYTE *)(v33 + 25) |= 1u;
    v41 = (__m128i *)v33;
    *(_BYTE *)(v33 + 58) |= 1u;
    *(_QWORD *)v33 = v44;
    v40 = v54[0];
    if ( v54[0] )
      goto LABEL_22;
    return sub_730620(a1, v41);
  }
  v40 = v54[0];
  v41 = (__m128i *)v33;
  if ( v54[0] )
  {
LABEL_22:
    v42 = (const __m128i *)sub_73DF90(v40, v41->m128i_i64);
    return sub_730620(a1, v42);
  }
  return sub_730620(a1, v41);
}
