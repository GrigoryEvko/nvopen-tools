// Function: sub_1D3CC80
// Address: 0x1d3cc80
//
__int64 __fastcall sub_1D3CC80(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  unsigned int v10; // r12d
  __int16 v12; // ax
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // r14
  unsigned int v18; // eax
  unsigned int v19; // r8d
  unsigned __int64 v20; // rax
  unsigned int v21; // edx
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r10
  __int64 v26; // r11
  __int64 v27; // rsi
  unsigned __int8 *v28; // r12
  const void **v29; // r15
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned int v33; // eax
  unsigned __int64 v34; // rcx
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // r15
  __int64 v38; // rax
  _QWORD *v39; // rsi
  __int64 v40; // rdx
  int v41; // eax
  __int64 v42; // rax
  _QWORD *v43; // r14
  unsigned int v44; // eax
  char v45; // cl
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rdx
  __int64 v49; // r8
  unsigned __int64 v50; // r9
  __int64 v51; // rsi
  __int64 v52; // r15
  unsigned __int8 *v53; // rax
  const void **v54; // r12
  __int64 v55; // rcx
  __int64 v56; // rsi
  unsigned __int8 *v57; // r12
  const void **v58; // r8
  unsigned int v59; // r15d
  __int128 v60; // [rsp-10h] [rbp-80h]
  int v61; // [rsp+4h] [rbp-6Ch]
  unsigned int v62; // [rsp+8h] [rbp-68h]
  __int64 v63; // [rsp+8h] [rbp-68h]
  __int64 v64; // [rsp+8h] [rbp-68h]
  __int64 v65; // [rsp+8h] [rbp-68h]
  __int64 v66; // [rsp+10h] [rbp-60h]
  __int64 v67; // [rsp+10h] [rbp-60h]
  __int64 v68; // [rsp+10h] [rbp-60h]
  __int64 v69; // [rsp+10h] [rbp-60h]
  __int64 v70; // [rsp+10h] [rbp-60h]
  const void **v71; // [rsp+10h] [rbp-60h]
  const void **v72; // [rsp+10h] [rbp-60h]
  __int64 v73; // [rsp+18h] [rbp-58h]
  unsigned __int64 v74; // [rsp+18h] [rbp-58h]
  unsigned __int64 v75; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v76; // [rsp+28h] [rbp-48h]
  unsigned __int64 v77; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v78; // [rsp+38h] [rbp-38h]

  v10 = a3;
  v12 = *(_WORD *)(a2 + 24);
  if ( v12 <= 120 )
  {
    if ( v12 > 118 )
    {
      if ( (unsigned __int8)sub_1D1F940(
                              (__int64)a1,
                              **(_QWORD **)(a2 + 32),
                              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                              a4,
                              0) )
        return *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
      if ( (unsigned __int8)sub_1D1F940(
                              (__int64)a1,
                              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                              a4,
                              0) )
        return **(_QWORD **)(a2 + 32);
      return 0;
    }
    if ( v12 != 10 )
    {
      if ( v12 == 118 )
      {
        v13 = sub_1D1ADA0(
                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                a3,
                a4,
                a5,
                a6);
        if ( v13 )
        {
          v14 = *(_QWORD *)(v13 + 88);
          if ( *(_DWORD *)(a4 + 8) <= 0x40u )
          {
            if ( (*(_QWORD *)a4 & ~*(_QWORD *)(v14 + 24)) == 0 )
              return **(_QWORD **)(a2 + 32);
          }
          else if ( (unsigned __int8)sub_16A5A00((__int64 *)a4, (__int64 *)(v14 + 24)) )
          {
            return **(_QWORD **)(a2 + 32);
          }
        }
      }
      return 0;
    }
    v32 = *(_QWORD *)(a2 + 88);
    v33 = *(_DWORD *)(v32 + 32);
    v78 = v33;
    if ( v33 > 0x40 )
    {
      v65 = v32;
      v71 = (const void **)(v32 + 24);
      sub_16A4FD0((__int64)&v77, (const void **)(v32 + 24));
      v33 = v78;
      if ( v78 > 0x40 )
      {
        sub_16A8890((__int64 *)&v77, (__int64 *)a4);
        v37 = v77;
        v76 = v78;
        v75 = v77;
        if ( v78 > 0x40 )
        {
          if ( !sub_16A5220((__int64)&v75, v71) )
            goto LABEL_54;
          if ( v37 )
            j_j___libc_free_0_0(v37);
          return 0;
        }
        v35 = *(_QWORD *)(v65 + 24);
LABEL_31:
        if ( v35 != v37 )
        {
LABEL_54:
          v56 = *(_QWORD *)(a2 + 72);
          v57 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * v10);
          v58 = (const void **)*((_QWORD *)v57 + 1);
          v59 = *v57;
          v77 = v56;
          if ( v56 )
          {
            v72 = v58;
            sub_1623A60((__int64)&v77, v56, 2);
            v58 = v72;
          }
          v78 = *(_DWORD *)(a2 + 64);
          result = sub_1D38970((__int64)a1, (__int64)&v75, (__int64)&v77, v59, v58, 0, a7, a8, a9, 0);
          v31 = v77;
          if ( !v77 )
            goto LABEL_23;
          goto LABEL_22;
        }
        return 0;
      }
      v34 = v77;
      v35 = *(_QWORD *)(v65 + 24);
    }
    else
    {
      v34 = *(_QWORD *)(v32 + 24);
      v35 = v34;
    }
    v36 = *(_QWORD *)a4 & v34;
    v76 = v33;
    v75 = v36;
    v37 = v36;
    goto LABEL_31;
  }
  if ( v12 != 124 )
  {
    if ( v12 != 144 )
      return 0;
    v16 = *(_QWORD *)(a2 + 32);
    v17 = *(_QWORD *)(v16 + 8);
    v66 = *(_QWORD *)v16;
    v18 = sub_1D142E0(*(_QWORD *)v16, *(_DWORD *)(v16 + 8));
    v19 = v18;
    if ( *(_DWORD *)(a4 + 8) > 0x40u )
    {
      v61 = *(_DWORD *)(a4 + 8);
      v62 = v18;
      v22 = sub_16A57B0(a4);
      v19 = v62;
      v21 = v61 - v22;
    }
    else
    {
      if ( !*(_QWORD *)a4 )
      {
LABEL_18:
        sub_16A5A50((__int64)&v75, (__int64 *)a4, v19);
        v23 = sub_1D3CC80(a1, v66, v17, &v75);
        v25 = v23;
        v26 = v24;
        if ( !v23 )
        {
          if ( v76 > 0x40 && v75 )
            j_j___libc_free_0_0(v75);
          return 0;
        }
        v27 = *(_QWORD *)(a2 + 72);
        v28 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * v10);
        v29 = (const void **)*((_QWORD *)v28 + 1);
        v30 = *v28;
        v77 = v27;
        if ( v27 )
        {
          v73 = v24;
          v63 = v30;
          v67 = v23;
          sub_1623A60((__int64)&v77, v27, 2);
          v30 = v63;
          v25 = v67;
          v26 = v73;
        }
        *((_QWORD *)&v60 + 1) = v26;
        *(_QWORD *)&v60 = v25;
        v78 = *(_DWORD *)(a2 + 64);
        result = sub_1D309E0(
                   a1,
                   144,
                   (__int64)&v77,
                   v30,
                   v29,
                   0,
                   *(double *)a7.m128i_i64,
                   a8,
                   *(double *)a9.m128i_i64,
                   v60);
        v31 = v77;
        if ( !v77 )
          goto LABEL_23;
        goto LABEL_22;
      }
      _BitScanReverse64(&v20, *(_QWORD *)a4);
      v21 = 64 - (v20 ^ 0x3F);
    }
    if ( v19 < v21 )
      return 0;
    goto LABEL_18;
  }
  v38 = *(_QWORD *)(a2 + 48);
  if ( !v38 )
    return 0;
  if ( *(_QWORD *)(v38 + 32) )
    return 0;
  v39 = *(_QWORD **)(a2 + 32);
  v40 = v39[5];
  v41 = *(unsigned __int16 *)(v40 + 24);
  if ( v41 != 32 && v41 != 10 )
    return 0;
  v42 = *(_QWORD *)(v40 + 88);
  v43 = *(_QWORD **)(v42 + 24);
  if ( *(_DWORD *)(v42 + 32) > 0x40u )
    v43 = (_QWORD *)*v43;
  v44 = *(_DWORD *)(a4 + 8);
  v45 = (char)v43;
  if ( (unsigned int)v43 >= v44 )
    return 0;
  v76 = v44;
  if ( v44 <= 0x40 )
  {
    v46 = *(_QWORD *)a4;
LABEL_42:
    v47 = v46 << v45;
    goto LABEL_43;
  }
  sub_16A4FD0((__int64)&v75, (const void **)a4);
  LOBYTE(v44) = v76;
  if ( v76 > 0x40 )
  {
    sub_16A7DC0((__int64 *)&v75, (unsigned int)v43);
    v39 = *(_QWORD **)(a2 + 32);
    goto LABEL_44;
  }
  v45 = (char)v43;
  v39 = *(_QWORD **)(a2 + 32);
  v47 = 0;
  if ( (_DWORD)v43 != v76 )
  {
    v46 = v75;
    goto LABEL_42;
  }
LABEL_43:
  v75 = v47 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v44);
LABEL_44:
  v49 = sub_1D3CC80(a1, *v39, v39[1], &v75);
  v50 = v48;
  if ( !v49 )
  {
    sub_135E100((__int64 *)&v75);
    return 0;
  }
  v51 = *(_QWORD *)(a2 + 72);
  v52 = *(_QWORD *)(a2 + 32);
  v53 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * v10);
  v54 = (const void **)*((_QWORD *)v53 + 1);
  v55 = *v53;
  v77 = v51;
  if ( v51 )
  {
    v74 = v48;
    v64 = v55;
    v70 = v49;
    sub_1623A60((__int64)&v77, v51, 2);
    v55 = v64;
    v49 = v70;
    v50 = v74;
  }
  v78 = *(_DWORD *)(a2 + 64);
  result = (__int64)sub_1D332F0(
                      a1,
                      124,
                      (__int64)&v77,
                      v55,
                      v54,
                      0,
                      *(double *)a7.m128i_i64,
                      a8,
                      a9,
                      v49,
                      v50,
                      *(_OWORD *)(v52 + 40));
  v31 = v77;
  if ( !v77 )
    goto LABEL_23;
LABEL_22:
  v68 = result;
  sub_161E7C0((__int64)&v77, v31);
  result = v68;
LABEL_23:
  if ( v76 > 0x40 && v75 )
  {
    v69 = result;
    j_j___libc_free_0_0(v75);
    return v69;
  }
  return result;
}
