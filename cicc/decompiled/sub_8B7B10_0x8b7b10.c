// Function: sub_8B7B10
// Address: 0x8b7b10
//
__int64 __fastcall sub_8B7B10(
        __m128i *a1,
        __m128i *a2,
        __int64 *a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8)
{
  __m128i *i; // rbx
  __int64 v13; // rdi
  __int64 v14; // r11
  __m128i **j; // rsi
  __m128i *v16; // rcx
  _QWORD *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // r15d
  __int64 v22; // r11
  __int64 v23; // r10
  unsigned __int8 *v24; // rdi
  __int64 **v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 *v29; // r9
  unsigned int v30; // r8d
  int v31; // eax
  _QWORD *v32; // r10
  __int64 v33; // r11
  _QWORD *v34; // rax
  __int64 v35; // rdx
  int v36; // eax
  __int64 v37; // r11
  int v38; // eax
  int v39; // eax
  __int64 **v40; // rax
  __int64 v41; // r11
  char v42; // r8
  _QWORD *v43; // rcx
  unsigned int v44; // eax
  __int64 v45; // r11
  unsigned int v46; // eax
  __m128i *v47; // rax
  __int64 *v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // r15
  unsigned __int8 v52; // [rsp+8h] [rbp-68h]
  __int64 v53; // [rsp+10h] [rbp-60h]
  _QWORD *v54; // [rsp+10h] [rbp-60h]
  __int64 v55; // [rsp+10h] [rbp-60h]
  _QWORD *v56; // [rsp+10h] [rbp-60h]
  __int64 v57; // [rsp+18h] [rbp-58h]
  _QWORD *v58; // [rsp+18h] [rbp-58h]
  __int64 v59; // [rsp+18h] [rbp-58h]
  __int64 v60; // [rsp+18h] [rbp-58h]
  __int64 v61; // [rsp+18h] [rbp-58h]
  __int64 v62; // [rsp+18h] [rbp-58h]
  __int64 v63; // [rsp+18h] [rbp-58h]
  __m128i *v64[2]; // [rsp+20h] [rbp-50h] BYREF
  int v65; // [rsp+30h] [rbp-40h]

  for ( i = a2; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
    ;
  v13 = i[10].m128i_i64[1];
  if ( a1[5].m128i_i8[0] == 10 )
    v14 = *(_QWORD *)(a1[6].m128i_i64[0] + 56);
  else
    v14 = a1[5].m128i_i64[1];
  *a3 = 0;
  for ( j = *(__m128i ***)(*(_QWORD *)(v14 + 176) + 152LL); *((_BYTE *)j + 140) == 12; j = (__m128i **)j[20] )
    ;
  v16 = j[21];
  v17 = *(_QWORD **)v13;
  v18 = v16->m128i_i64[0];
  if ( *(_QWORD *)v13 )
  {
    while ( v18 )
    {
      if ( (*(_BYTE *)(v18 + 33) & 1) == 0 )
        v18 = *(_QWORD *)v18;
      v17 = (_QWORD *)*v17;
      if ( !v17 )
        goto LABEL_15;
    }
    goto LABEL_13;
  }
LABEL_15:
  if ( v18 && (*(_BYTE *)(v18 + 33) & 1) == 0 || (v57 = v14, ((v16[1].m128i_i8[0] ^ *(_BYTE *)(v13 + 16)) & 1) != 0) )
  {
LABEL_13:
    v19 = 0;
    v20 = 0;
    goto LABEL_14;
  }
  sub_865900((__int64)a1);
  v22 = v57;
  v23 = a5;
  if ( a6 )
  {
    v24 = (unsigned __int8 *)a1;
    v25 = sub_8B1C20((__int64)a1, a6, v64, 0, 0x20000u);
    v22 = v57;
    v23 = a5;
    j = (__m128i **)v25;
    *a3 = (__int64)v64[0]->m128i_i64;
    if ( !v25 )
      goto LABEL_29;
  }
  v30 = 80;
  if ( a7 )
    v30 = a8 == 0 ? 18 : 82;
  v24 = (unsigned __int8 *)i;
  v53 = v22;
  v58 = (_QWORD *)v23;
  v31 = sub_8B3500(i, (__int64)j, a3, v23, v30);
  v32 = v58;
  v33 = v53;
  if ( !v31 )
    goto LABEL_29;
  v34 = *(_QWORD **)(v53 + 104);
  if ( v34 )
  {
    v35 = v34[22];
    if ( v35 )
    {
      if ( *(_QWORD *)(v35 + 16) || (*(_BYTE *)(*(_QWORD *)(*v34 + 88LL) + 160LL) & 0x20) != 0 )
      {
        v24 = (unsigned __int8 *)*a3;
        j = (__m128i **)v58;
        v36 = sub_8AF210((__m128i *)*a3, v58, 0x20000u, (__int64)a1, v53, 0);
        v37 = v53;
        if ( !v36 )
          goto LABEL_29;
        v54 = v58;
        v59 = v37;
        v38 = sub_89A370((__int64 *)*a3);
        v33 = v59;
        v32 = v54;
        if ( !v38 )
        {
          j = (__m128i **)*a3;
          v24 = (unsigned __int8 *)a1;
          v39 = sub_8A00C0((__int64)a1, (__int64 *)*a3, 0);
          v33 = v59;
          v32 = v54;
          if ( !v39 )
            goto LABEL_29;
        }
      }
    }
  }
  j = (__m128i **)a1;
  v24 = (unsigned __int8 *)a3;
  v60 = v33;
  v40 = sub_8B2240(a3, (__int64)a1, v32, 0x20000u, 0);
  v41 = v60;
  if ( !v40 )
    goto LABEL_29;
  if ( a7 )
  {
    j = (__m128i **)v40;
    v24 = (unsigned __int8 *)i;
    v46 = sub_8D97D0(i, v40, a8 == 0 ? 1154 : 9346, v27, v28);
    v45 = v60;
    v28 = 1;
    v27 = 0;
    v20 = v46;
  }
  else
  {
    v42 = 1;
    v43 = 0;
    if ( (*(_BYTE *)(*(_QWORD *)(v60 + 176) + 207LL) & 0x10) != 0 )
    {
      v49 = sub_8B74F0(
              (unsigned __int64)a1,
              (__int64 ***)a3,
              a6 != 0,
              dword_4F07508,
              (__int64)dword_4F07508,
              (__int64)v29);
      v41 = v60;
      v50 = v49[11];
      v43 = v49;
      if ( (*(_BYTE *)(v50 + 207) & 0x10) != 0 )
      {
        v56 = v49;
        sub_8B1A30(v49[11], (FILE *)dword_4F07508);
        v43 = v56;
        v41 = v60;
      }
      v40 = *(__int64 ***)(v50 + 152);
      v42 = 0;
    }
    j = (__m128i **)i;
    v24 = (unsigned __int8 *)v40;
    v52 = v42;
    v55 = (__int64)v43;
    v61 = v41;
    v44 = sub_8DED30(v40, i, a8 == 0 ? 34960 : 1083536);
    v28 = v52;
    v27 = v55;
    v45 = v61;
    v20 = v44;
  }
  if ( !v20 )
  {
LABEL_29:
    sub_864110((__int64)v24, (__int64)j, v26, v27, v28, v29);
    if ( *a3 )
    {
      sub_725130((__int64 *)*a3);
      *a3 = 0;
    }
    goto LABEL_13;
  }
  if ( (_BYTE)v28 )
  {
    v47 = (__m128i *)*a3;
    v64[0] = a1;
    v65 = 0;
    v64[1] = v47;
    v24 = *(unsigned __int8 **)(v45 + 136);
    if ( !v24 )
    {
      v63 = v45;
      v24 = (unsigned __int8 *)sub_881A70(0, 0xBu, 12, 13, v28, (__int64)v29);
      *(_QWORD *)(v63 + 136) = v24;
    }
    j = v64;
    v48 = (__int64 *)sub_881B20(v24, (__int64)v64, 0);
    v27 = (__int64)v48;
    if ( v48 )
      v27 = *v48;
  }
  v62 = v27;
  sub_864110((__int64)v24, (__int64)j, v26, v27, v28, v29);
  v19 = v62;
LABEL_14:
  *a4 = v19;
  return v20;
}
