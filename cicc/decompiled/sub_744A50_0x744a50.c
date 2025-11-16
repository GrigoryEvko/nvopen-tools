// Function: sub_744A50
// Address: 0x744a50
//
__int64 __fastcall sub_744A50(
        __m128i *a1,
        __m128i *a2,
        __int64 a3,
        _QWORD *a4,
        _DWORD *a5,
        unsigned int a6,
        int *a7,
        __m128i *a8)
{
  __int64 v13; // rsi
  int v14; // eax
  _QWORD *v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rcx
  _UNKNOWN *__ptr32 *v18; // r8
  __int64 *v20; // rax
  __int64 v21; // rdx
  int v22; // eax
  _QWORD *v23; // rcx
  bool v24; // zf
  __int64 *v25; // r14
  _QWORD *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r9
  __int64 v29; // rsi
  unsigned __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rdi
  int v33; // eax
  __int64 v34; // r11
  int v35; // eax
  int v36; // eax
  __int64 *v37; // rax
  int v38; // eax
  __int64 *v39; // rdi
  __int64 v40; // r9
  _QWORD *v41; // rax
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rdi
  const __m128i *v58; // rdi
  __int64 v59; // [rsp-8h] [rbp-78h]
  __m128i *v60; // [rsp+0h] [rbp-70h]
  _QWORD *v62; // [rsp+8h] [rbp-68h]
  __m128i *v63; // [rsp+8h] [rbp-68h]
  int v64; // [rsp+1Ch] [rbp-54h] BYREF
  int v65; // [rsp+20h] [rbp-50h] BYREF
  _BYTE v66[4]; // [rsp+24h] [rbp-4Ch] BYREF
  __int64 v67; // [rsp+28h] [rbp-48h] BYREF
  __m128i *v68; // [rsp+30h] [rbp-40h] BYREF
  __m128i *v69; // [rsp+38h] [rbp-38h] BYREF

  v67 = 0;
  v69 = (__m128i *)sub_724DC0();
  sub_7296C0(&v65);
  v13 = (__int64)&v68;
  v14 = sub_72E9D0(a1, &v68, &v64);
  v15 = a4;
  if ( v14 )
  {
    v20 = sub_72E9A0((__int64)a1);
    if ( !v20[10] )
    {
      v13 = (__int64)v68;
      v16 = sub_744640((__int64)a1, v68, v64, a2, a3, a5, a6, a7, a8, v69);
      v67 = v16;
      goto LABEL_5;
    }
    v15 = a4;
    if ( a4 )
      goto LABEL_3;
    v15 = (_QWORD *)*v20;
  }
  if ( v15 )
  {
LABEL_3:
    if ( a1[10].m128i_i8[13] == 12 && a1[11].m128i_i8[0] == 1 )
    {
      v62 = v15;
      v60 = (__m128i *)sub_72E9A0((__int64)a1);
      v22 = sub_8D32E0(v62);
      v23 = v62;
      if ( v22 )
      {
        v26 = sub_7432A0(v60, a2, a3, v62, a5, a6, a7, a8->m128i_i64);
        v28 = (unsigned int)*a7;
        if ( (_DWORD)v28 )
        {
LABEL_23:
          v67 = (__int64)a1;
          goto LABEL_6;
        }
        if ( (*((_BYTE *)v26 + 25) & 3) != 0 )
        {
          v13 = (__int64)v69;
          if ( (unsigned int)sub_717510(v26, v69, 1, v27, v59, v28) )
          {
            v41 = (_QWORD *)sub_8D46C0(v69[8].m128i_i64[0]);
            v42 = sub_72D600(v41);
            v21 = (__int64)v69;
            v69[8].m128i_i64[0] = v42;
            v36 = *a7;
LABEL_33:
            if ( v36 )
              goto LABEL_23;
LABEL_18:
            if ( v67 )
              goto LABEL_6;
            goto LABEL_11;
          }
        }
LABEL_22:
        *a7 = 1;
        goto LABEL_23;
      }
      v24 = v60[1].m128i_i8[8] == 0;
      v67 = 0;
      if ( v24 )
      {
        v25 = sub_7305B0();
        goto LABEL_16;
      }
      v29 = (__int64)a2;
      v63 = v69;
      v30 = sub_7410C0(v60, a2, a3, v23, a5, a6, a7, a8->m128i_i64, v69, &v67);
      v25 = (__int64 *)v30;
      if ( !v30 )
      {
        if ( v63 )
        {
          if ( v63[10].m128i_i8[13] == 12 && v63[11].m128i_i8[0] == 1 )
          {
            v57 = v63[11].m128i_i64[1];
            if ( (*(_BYTE *)(v57 + 25) & 3) != 0 && dword_4F04C44 == -1 )
            {
              v21 = (__int64)qword_4F04C68;
              if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
              {
                v58 = (const __m128i *)sub_7196D0(v57, v29, (__int64)qword_4F04C68, v17, (__int64)v18, v31);
                if ( v58 )
                {
                  v13 = (__int64)v63;
                  sub_72A510(v58, v63);
                  if ( !*a7 )
                    goto LABEL_18;
                  goto LABEL_38;
                }
              }
            }
          }
        }
        goto LABEL_37;
      }
      v32 = *(_QWORD *)v30;
      if ( (*(_BYTE *)(v30 + 25) & 1) == 0 )
      {
        v33 = sub_8DBE70(v32);
        v34 = (__int64)v63;
        if ( v33 || (v13 = (__int64)v63, v35 = sub_716120((__int64)v25, (__int64)v63), v34 = (__int64)v63, !v35) )
        {
          v13 = v34;
          sub_70FD90(v25, v34);
        }
        if ( !*a7 )
        {
LABEL_31:
          if ( !*((_BYTE *)v25 + 24) )
          {
            sub_72C970((__int64)v69);
            v36 = *a7;
            goto LABEL_33;
          }
          if ( (unsigned int)sub_8D2E30(*v25) )
          {
            v13 = (__int64)v69;
            if ( (unsigned int)sub_717520(v25, (__int64)v69, 1) )
              goto LABEL_49;
          }
          v38 = sub_8DBE70(*v25);
          v13 = (__int64)v69;
          v39 = v25;
          if ( v38 )
            goto LABEL_48;
          if ( (unsigned int)sub_716120((__int64)v25, (__int64)v69) )
          {
LABEL_49:
            v36 = *a7;
            goto LABEL_33;
          }
          if ( (a6 & 4) != 0 || (unsigned int)sub_731EE0((__int64)v25, v13, v21, v17, (__int64)v18, v40) )
          {
            v13 = (__int64)v69;
            v39 = v25;
LABEL_48:
            sub_70FD90(v39, v13);
            goto LABEL_49;
          }
          sub_72C970((__int64)v69);
          goto LABEL_22;
        }
LABEL_38:
        v37 = sub_7305B0();
        v67 = 0;
        v25 = v37;
        v36 = *a7;
        if ( !v25 )
          goto LABEL_33;
        if ( v36 )
          goto LABEL_23;
        goto LABEL_31;
      }
      if ( (unsigned int)sub_8D3410(v32) )
      {
        v13 = (__int64)v63;
        if ( (unsigned int)sub_717510(v25, v63, 1, v43, v44, v45) )
        {
          v46 = sub_8D67C0(*v25);
LABEL_55:
          sub_70FEE0((__int64)v63, v46, v47, v48, v49);
LABEL_37:
          v13 = (unsigned int)*a7;
          if ( !(_DWORD)v13 )
            goto LABEL_18;
          goto LABEL_38;
        }
        v25 = (__int64 *)sub_6EE5A0((__int64)v25);
      }
      else if ( (unsigned int)sub_8D2310(*v25) )
      {
        v13 = (__int64)v63;
        if ( (unsigned int)sub_717510(v25, v63, 1, v50, v51, v52) )
        {
          v46 = sub_72D2E0((_QWORD *)*v25);
          goto LABEL_55;
        }
        v25 = sub_731370((__int64)v25, (__int64)v63, v53, v54, v55, v56);
      }
      else
      {
        v13 = (__int64)v66;
        v25 = sub_6EE530((__int64)v25, (unsigned __int64)v66, &v67, (__int64)a5, a6);
      }
LABEL_16:
      v17 = (unsigned int)*a7;
      if ( !(_DWORD)v17 )
      {
        if ( !v25 )
          goto LABEL_18;
        goto LABEL_31;
      }
      goto LABEL_38;
    }
  }
  v13 = (__int64)a2;
  v16 = (__int64)sub_743600(a1, a2, a3, v15, a5, a6, a7, a8, v69);
  v67 = v16;
LABEL_5:
  if ( !v16 )
  {
    v21 = (unsigned int)*a7;
    if ( !(_DWORD)v21 )
LABEL_11:
      v67 = sub_73A460(v69, v13, v21, v17, v18);
  }
LABEL_6:
  sub_729730(v65);
  sub_724E30((__int64)&v69);
  return v67;
}
