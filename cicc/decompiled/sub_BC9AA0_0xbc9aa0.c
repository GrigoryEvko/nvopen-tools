// Function: sub_BC9AA0
// Address: 0xbc9aa0
//
__int64 __fastcall sub_BC9AA0(_BYTE *a1)
{
  unsigned __int8 v2; // al
  _BYTE **v3; // r14
  _BYTE *v4; // r12
  _BYTE *v5; // rdi
  int v6; // r13d
  int v7; // eax
  unsigned __int8 v8; // al
  _BYTE *v9; // rdx
  _BYTE *v10; // rdi
  unsigned __int8 v11; // al
  _BYTE *v12; // rdx
  _BYTE *v13; // rdi
  unsigned __int8 v14; // al
  _BYTE *v15; // rdx
  _BYTE *v16; // rdi
  unsigned __int8 v17; // al
  _BYTE *v18; // rdx
  _BYTE *v19; // rdi
  unsigned __int8 v20; // al
  _BYTE *v21; // rdx
  _BYTE *v22; // rdi
  unsigned __int8 v23; // al
  _BYTE *v24; // rdx
  _BYTE *v25; // rdi
  unsigned __int8 v26; // al
  _BYTE *v27; // rdx
  _BYTE *v28; // rdi
  bool v29; // zf
  unsigned __int8 v30; // al
  __int64 v31; // r15
  int v32; // r14d
  _BYTE *v33; // rcx
  _BYTE *v34; // rdi
  int v35; // edx
  _BYTE *v36; // rax
  __int64 v37; // r14
  double v38; // xmm0_8
  bool v39; // dl
  unsigned int v40; // ecx
  _BYTE *v41; // r12
  _BYTE *v42; // rbx
  unsigned __int8 v43; // al
  __int64 *v44; // rdx
  _BYTE *v45; // r12
  __int64 v46; // rax
  __int64 v47; // rdx
  const __m128i *v48; // rdi
  __int64 v49; // r12
  int v51; // eax
  unsigned __int8 v52; // al
  _BYTE *v53; // r12
  _BYTE *v54; // rax
  unsigned __int8 v55; // dl
  _QWORD *v56; // rbx
  __int64 v57; // rcx
  _QWORD *v58; // r12
  _BYTE *v59; // rax
  unsigned __int8 v60; // dl
  __int64 *v61; // rax
  __int64 v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdi
  __int64 v67; // rdx
  _QWORD *v68; // rax
  __int64 v69; // rcx
  _QWORD *v70; // rdx
  const __m128i *v71; // rsi
  __int64 v72; // rax
  const __m128i *v73; // r11
  int v74; // r14d
  __int64 v75; // rcx
  __int64 v76; // rsi
  bool v77; // bl
  int v78; // r15d
  __int64 v79; // rdx
  __int64 v80; // r8
  unsigned __int64 v81; // r13
  __int64 v82; // rax
  __m128i *v83; // r9
  __m128i *v84; // r10
  const __m128i *v85; // rax
  __int64 v86; // [rsp+10h] [rbp-C0h]
  __int64 v87; // [rsp+20h] [rbp-B0h]
  __int64 v88; // [rsp+28h] [rbp-A8h]
  __int64 v89; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v90; // [rsp+38h] [rbp-98h] BYREF
  __int64 v91; // [rsp+40h] [rbp-90h] BYREF
  __int64 v92; // [rsp+48h] [rbp-88h] BYREF
  __int64 v93; // [rsp+50h] [rbp-80h] BYREF
  __int64 v94; // [rsp+58h] [rbp-78h] BYREF
  __int64 v95; // [rsp+60h] [rbp-70h] BYREF
  __int64 v96; // [rsp+68h] [rbp-68h] BYREF
  _QWORD *v97; // [rsp+70h] [rbp-60h] BYREF
  _QWORD *v98; // [rsp+78h] [rbp-58h] BYREF
  const __m128i *v99; // [rsp+80h] [rbp-50h] BYREF
  const __m128i *v100; // [rsp+88h] [rbp-48h]
  const __m128i *v101; // [rsp+90h] [rbp-40h]

  if ( !a1 || *a1 != 5 )
    return 0;
  v2 = *(a1 - 16);
  if ( (v2 & 2) != 0 )
  {
    if ( (unsigned int)(*((_DWORD *)a1 - 6) - 8) > 2 )
      return 0;
    v3 = (_BYTE **)*((_QWORD *)a1 - 4);
    v4 = a1 - 16;
  }
  else
  {
    if ( ((*((_WORD *)a1 - 8) >> 6) & 0xFu) - 8 > 2 )
      return 0;
    v4 = a1 - 16;
    v3 = (_BYTE **)&a1[-8 * ((v2 >> 2) & 0xF) - 16];
  }
  v5 = *v3;
  if ( !*v3 )
    return 0;
  if ( *v5 == 5 )
  {
    v6 = (*(v5 - 16) & 2) != 0 ? *((_DWORD *)v5 - 6) : (*((_WORD *)v5 - 8) >> 6) & 0xF;
    if ( v6 != 2 )
      goto LABEL_11;
    if ( sub_BC97F0((__int64)v5, "SampleProfile") )
      goto LABEL_15;
    v5 = *v3;
    if ( !*v3 )
      return 0;
    if ( *v5 == 5 )
    {
LABEL_11:
      if ( (*(v5 - 16) & 2) != 0 )
        v7 = *((_DWORD *)v5 - 6);
      else
        v7 = (*((_WORD *)v5 - 8) >> 6) & 0xF;
      if ( v7 == 2 )
      {
        v6 = 0;
        if ( sub_BC97F0((__int64)v5, "InstrProf") )
          goto LABEL_15;
        v5 = *v3;
        if ( !*v3 )
          return 0;
      }
    }
  }
  if ( *v5 != 5 )
    return 0;
  v51 = (*(v5 - 16) & 2) != 0 ? *((_DWORD *)v5 - 6) : (*((_WORD *)v5 - 8) >> 6) & 0xF;
  if ( v51 != 2 || !sub_BC97F0((__int64)v5, "CSInstrProf") )
    return 0;
  v6 = 1;
LABEL_15:
  v8 = *(a1 - 16);
  if ( (v8 & 2) != 0 )
    v9 = (_BYTE *)*((_QWORD *)a1 - 4);
  else
    v9 = &v4[-8 * ((v8 >> 2) & 0xF)];
  v10 = (_BYTE *)*((_QWORD *)v9 + 1);
  if ( *v10 != 5 )
    v10 = 0;
  if ( !(unsigned __int8)sub_BC9770((__int64)v10, "TotalCount", &v90) )
    return 0;
  v11 = *(a1 - 16);
  if ( (v11 & 2) != 0 )
    v12 = (_BYTE *)*((_QWORD *)a1 - 4);
  else
    v12 = &v4[-8 * ((v11 >> 2) & 0xF)];
  v13 = (_BYTE *)*((_QWORD *)v12 + 2);
  if ( *v13 != 5 )
    v13 = 0;
  if ( !(unsigned __int8)sub_BC9770((__int64)v13, "MaxCount", &v93) )
    return 0;
  v14 = *(a1 - 16);
  if ( (v14 & 2) != 0 )
    v15 = (_BYTE *)*((_QWORD *)a1 - 4);
  else
    v15 = &v4[-8 * ((v14 >> 2) & 0xF)];
  v16 = (_BYTE *)*((_QWORD *)v15 + 3);
  if ( *v16 != 5 )
    v16 = 0;
  if ( !(unsigned __int8)sub_BC9770((__int64)v16, "MaxInternalCount", &v94) )
    return 0;
  v17 = *(a1 - 16);
  if ( (v17 & 2) != 0 )
    v18 = (_BYTE *)*((_QWORD *)a1 - 4);
  else
    v18 = &v4[-8 * ((v17 >> 2) & 0xF)];
  v19 = (_BYTE *)*((_QWORD *)v18 + 4);
  if ( *v19 != 5 )
    v19 = 0;
  if ( !(unsigned __int8)sub_BC9770((__int64)v19, "MaxFunctionCount", &v92) )
    return 0;
  v20 = *(a1 - 16);
  if ( (v20 & 2) != 0 )
    v21 = (_BYTE *)*((_QWORD *)a1 - 4);
  else
    v21 = &v4[-8 * ((v20 >> 2) & 0xF)];
  v22 = (_BYTE *)*((_QWORD *)v21 + 5);
  if ( *v22 != 5 )
    v22 = 0;
  if ( !(unsigned __int8)sub_BC9770((__int64)v22, "NumCounts", &v89) )
    return 0;
  v23 = *(a1 - 16);
  if ( (v23 & 2) != 0 )
    v24 = (_BYTE *)*((_QWORD *)a1 - 4);
  else
    v24 = &v4[-8 * ((v23 >> 2) & 0xF)];
  v25 = (_BYTE *)*((_QWORD *)v24 + 6);
  if ( *v25 != 5 )
    v25 = 0;
  if ( !(unsigned __int8)sub_BC9770((__int64)v25, "NumFunctions", &v91) )
    return 0;
  v26 = *(a1 - 16);
  v95 = 0;
  if ( (v26 & 2) != 0 )
    v27 = (_BYTE *)*((_QWORD *)a1 - 4);
  else
    v27 = &v4[-8 * ((v26 >> 2) & 0xF)];
  v28 = (_BYTE *)*((_QWORD *)v27 + 7);
  if ( *v28 != 5 )
    v28 = 0;
  v29 = (unsigned __int8)sub_BC9770((__int64)v28, "IsPartialProfile", &v95) == 0;
  v30 = *(a1 - 16);
  if ( !v29 )
  {
    if ( (v30 & 2) != 0 )
    {
      if ( *((_DWORD *)a1 - 6) <= 8u )
        return 0;
      v31 = 64;
      v32 = 8;
      goto LABEL_53;
    }
    if ( ((*((_WORD *)a1 - 8) >> 6) & 0xFu) <= 8 )
      return 0;
    v31 = 64;
    v32 = 8;
LABEL_101:
    v33 = &v4[-8 * ((v30 >> 2) & 0xF)];
    goto LABEL_54;
  }
  v31 = 56;
  v32 = 7;
  if ( (v30 & 2) == 0 )
    goto LABEL_101;
LABEL_53:
  v33 = (_BYTE *)*((_QWORD *)a1 - 4);
LABEL_54:
  v34 = *(_BYTE **)&v33[v31];
  if ( *v34 != 5 )
    goto LABEL_98;
  v35 = (*(v34 - 16) & 2) != 0 ? *((_DWORD *)v34 - 6) : (*((_WORD *)v34 - 8) >> 6) & 0xF;
  if ( v35 != 2 )
    goto LABEL_98;
  v36 = sub_BC96B0((__int64)v34, "PartialProfileRatio");
  if ( !v36 )
  {
    v30 = *(a1 - 16);
LABEL_98:
    v38 = 0.0;
    v39 = (v30 & 2) != 0;
    goto LABEL_63;
  }
  v37 = (unsigned int)(v32 + 1);
  v38 = sub_C41B00(*((_QWORD *)v36 + 17) + 24LL);
  v30 = *(a1 - 16);
  v39 = (v30 & 2) != 0;
  if ( (v30 & 2) != 0 )
    v40 = *((_DWORD *)a1 - 6);
  else
    v40 = (*((_WORD *)a1 - 8) >> 6) & 0xF;
  if ( (unsigned int)v37 >= v40 )
    return 0;
  v31 = 8 * v37;
LABEL_63:
  v99 = 0;
  v100 = 0;
  v101 = 0;
  if ( v39 )
    v41 = (_BYTE *)*((_QWORD *)a1 - 4);
  else
    v41 = &v4[-8 * ((v30 >> 2) & 0xF)];
  v42 = *(_BYTE **)&v41[v31];
  if ( *v42 != 5 )
    return 0;
  v43 = *(v42 - 16);
  if ( (v43 & 2) != 0 )
  {
    if ( *((_DWORD *)v42 - 6) == 2 )
    {
      v44 = (__int64 *)*((_QWORD *)v42 - 4);
      v45 = v42 - 16;
      goto LABEL_69;
    }
    return 0;
  }
  if ( ((*((_WORD *)v42 - 8) >> 6) & 0xF) != 2 )
    return 0;
  v45 = v42 - 16;
  v44 = (__int64 *)&v42[-8 * ((v43 >> 2) & 0xF) - 16];
LABEL_69:
  if ( *(_BYTE *)*v44 )
    return 0;
  v46 = sub_B91420(*v44);
  if ( v47 == 15
    && *(_QWORD *)v46 == 0x64656C6961746544LL
    && *(_DWORD *)(v46 + 8) == 1835890003
    && *(_WORD *)(v46 + 12) == 29281
    && *(_BYTE *)(v46 + 14) == 121 )
  {
    v52 = *(v42 - 16);
    v53 = (v52 & 2) != 0 ? (_BYTE *)*((_QWORD *)v42 - 4) : &v45[-8 * ((v52 >> 2) & 0xF)];
    v54 = (_BYTE *)*((_QWORD *)v53 + 1);
    if ( *v54 == 5 )
    {
      v55 = *(v54 - 16);
      if ( (v55 & 2) != 0 )
      {
        v56 = (_QWORD *)*((_QWORD *)v54 - 4);
        v57 = *((unsigned int *)v54 - 6);
      }
      else
      {
        v57 = (*((_WORD *)v54 - 8) >> 6) & 0xF;
        v56 = &v54[-8 * ((v55 >> 2) & 0xF) - 16];
      }
      v58 = &v56[v57];
      if ( v58 == v56 )
      {
LABEL_140:
        v72 = sub_22077B0(88);
        v49 = v72;
        if ( v72 )
        {
          v73 = v100;
          v48 = v99;
          *(_DWORD *)v72 = v6;
          v29 = v95 == 0;
          v74 = v91;
          *(_QWORD *)(v72 + 8) = 0;
          v75 = v94;
          v76 = v93;
          v77 = !v29;
          *(_QWORD *)(v72 + 16) = 0;
          v78 = v89;
          *(_QWORD *)(v72 + 24) = 0;
          v79 = v92;
          v80 = v90;
          v81 = (char *)v73 - (char *)v48;
          if ( v73 == v48 )
          {
            v83 = 0;
          }
          else
          {
            if ( v81 > 0x7FFFFFFFFFFFFFF8LL )
              sub_4261EA(v48, v76, v92, v75);
            v86 = v90;
            v87 = v75;
            v88 = v92;
            v82 = sub_22077B0((char *)v73 - (char *)v48);
            v73 = v100;
            v48 = v99;
            v79 = v88;
            v75 = v87;
            v83 = (__m128i *)v82;
            v80 = v86;
          }
          *(_QWORD *)(v49 + 8) = v83;
          *(_QWORD *)(v49 + 16) = v83;
          *(_QWORD *)(v49 + 24) = (char *)v83 + v81;
          if ( v48 != v73 )
          {
            v84 = v83;
            v85 = v48;
            do
            {
              if ( v84 )
              {
                *v84 = _mm_loadu_si128(v85);
                v84[1].m128i_i64[0] = v85[1].m128i_i64[0];
              }
              v85 = (const __m128i *)((char *)v85 + 24);
              v84 = (__m128i *)((char *)v84 + 24);
            }
            while ( v85 != v73 );
            v83 = (__m128i *)((char *)v83
                            + 8 * ((unsigned __int64)((char *)&v85[-2].m128i_u64[1] - (char *)v48) >> 3)
                            + 24);
          }
          *(_QWORD *)(v49 + 16) = v83;
          *(_QWORD *)(v49 + 32) = v80;
          *(_QWORD *)(v49 + 40) = v76;
          *(_QWORD *)(v49 + 48) = v75;
          *(_QWORD *)(v49 + 56) = v79;
          *(_DWORD *)(v49 + 64) = v78;
          *(_DWORD *)(v49 + 68) = v74;
          *(_BYTE *)(v49 + 72) = v77;
          *(double *)(v49 + 80) = v38;
        }
        else
        {
          v48 = v99;
        }
        goto LABEL_73;
      }
      while ( 1 )
      {
        v59 = (_BYTE *)*v56;
        if ( *(_BYTE *)*v56 != 5 )
          break;
        v60 = *(v59 - 16);
        if ( (v60 & 2) != 0 )
        {
          if ( *((_DWORD *)v59 - 6) != 3 )
            break;
          v61 = (__int64 *)*((_QWORD *)v59 - 4);
        }
        else
        {
          if ( ((*((_WORD *)v59 - 8) >> 6) & 0xF) != 3 )
            break;
          v61 = (__int64 *)&v59[-8 * ((v60 >> 2) & 0xF) - 16];
        }
        v62 = *v61;
        v63 = v61[1];
        v64 = v61[2];
        if ( *(_BYTE *)v62 != 1 )
          v62 = 0;
        if ( *(_BYTE *)v63 != 1 )
          v63 = 0;
        if ( *(_BYTE *)v64 != 1 || !v62 || !v63 )
          break;
        v65 = *(_QWORD *)(v64 + 136);
        v66 = *(_QWORD *)(v65 + 24);
        if ( *(_DWORD *)(v65 + 32) > 0x40u )
          v66 = **(_QWORD **)(v65 + 24);
        v67 = *(_QWORD *)(v63 + 136);
        v96 = v66;
        v68 = *(_QWORD **)(v67 + 24);
        if ( *(_DWORD *)(v67 + 32) > 0x40u )
          v68 = (_QWORD *)*v68;
        v69 = *(_QWORD *)(v62 + 136);
        v97 = v68;
        v70 = *(_QWORD **)(v69 + 24);
        if ( *(_DWORD *)(v69 + 32) > 0x40u )
          v70 = (_QWORD *)*v70;
        v98 = v70;
        v71 = v100;
        if ( v100 == v101 )
        {
          sub_BC98E0(&v99, v100, &v98, &v97, &v96);
        }
        else
        {
          if ( v100 )
          {
            v100->m128i_i32[0] = (int)v70;
            v71->m128i_i64[1] = (__int64)v68;
            v71[1].m128i_i64[0] = v66;
            v71 = v100;
          }
          v100 = (const __m128i *)((char *)v71 + 24);
        }
        if ( v58 == ++v56 )
          goto LABEL_140;
      }
    }
  }
  v48 = v99;
  v49 = 0;
LABEL_73:
  if ( v48 )
    j_j___libc_free_0(v48, (char *)v101 - (char *)v48);
  return v49;
}
