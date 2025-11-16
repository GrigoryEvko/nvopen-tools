// Function: sub_291AEB0
// Address: 0x291aeb0
//
__int64 __fastcall sub_291AEB0(_BYTE *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, const __m128i *a6)
{
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r13
  __int64 *v8; // r12
  __int64 v10; // r13
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 **v20; // r12
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _BYTE *v24; // r12
  __int64 v25; // rax
  unsigned __int8 *v26; // r14
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rcx
  __int64 v31; // rdi
  __int64 (__fastcall *v32)(__int64, _BYTE *, unsigned __int8 *, __int64, __int64); // rax
  _QWORD *v33; // rax
  __int64 v34; // rbx
  __int64 v35; // r12
  __int64 v36; // rdx
  unsigned int v37; // esi
  __int64 v38; // rax
  __int64 v39; // rdi
  unsigned __int8 *v40; // r11
  __int64 (__fastcall *v41)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8); // rax
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // r14
  __int64 v47; // r15
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 v50; // r12
  __int64 v51; // rbx
  __int64 v52; // r12
  __int64 v53; // rdx
  unsigned int v54; // esi
  __int64 v55; // rax
  __int64 v56; // [rsp-8h] [rbp-E8h]
  unsigned __int64 v57; // [rsp+0h] [rbp-E0h]
  unsigned __int8 *v58; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v59; // [rsp+8h] [rbp-D8h]
  unsigned __int8 *v60; // [rsp+8h] [rbp-D8h]
  __int64 v62; // [rsp+18h] [rbp-C8h]
  __m128i v63[2]; // [rsp+20h] [rbp-C0h] BYREF
  char v64; // [rsp+40h] [rbp-A0h]
  char v65; // [rsp+41h] [rbp-9Fh]
  __m128i v66[2]; // [rsp+50h] [rbp-90h] BYREF
  char v67; // [rsp+70h] [rbp-70h]
  char v68; // [rsp+71h] [rbp-6Fh]
  __m128i v69[2]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v70; // [rsp+A0h] [rbp-40h]

  v6 = a3;
  v7 = a5;
  v8 = (__int64 *)a4;
  v62 = *(_QWORD *)(a3 + 8);
  if ( (_BYTE)qword_5005508 )
  {
    v15 = sub_9208B0((__int64)a1, a4);
    v69[0].m128i_i64[1] = v16;
    v69[0].m128i_i64[0] = (unsigned __int64)(v15 + 7) >> 3;
    v57 = sub_CA1930(v69);
    v17 = sub_9208B0((__int64)a1, v62);
    v69[0].m128i_i64[1] = v18;
    v69[0].m128i_i64[0] = (unsigned __int64)(v17 + 7) >> 3;
    a5 = sub_CA1930(v69);
    if ( a5 == 2 * v57 && (!v7 || v57 == v7) )
    {
      v19 = sub_BCDA70(v8, 2);
      v68 = 1;
      v20 = (__int64 **)v19;
      v67 = 3;
      v66[0].m128i_i64[0] = (__int64)".castvec";
      sub_9C6370(v69, a6, v66, v21, v22, v23);
      v24 = (_BYTE *)sub_291AC80((__int64 *)a2, 0x31u, v6, v20, (__int64)v69, 0, v63[0].m128i_i32[0], 0);
      v25 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
      v26 = (unsigned __int8 *)sub_ACD640(v25, (unsigned int)(v7 / v57), 0);
      v65 = 1;
      v63[0].m128i_i64[0] = (__int64)".extract";
      v64 = 3;
      sub_9C6370(v66, a6, v63, v27, v28, v29);
      v31 = *(_QWORD *)(a2 + 80);
      v32 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *, __int64, __int64))(*(_QWORD *)v31 + 96LL);
      if ( (char *)v32 == (char *)sub_948070 )
      {
        if ( *v24 > 0x15u || *v26 > 0x15u )
        {
LABEL_21:
          v70 = 257;
          v33 = sub_BD2C40(72, 2u);
          v11 = (__int64)v33;
          if ( v33 )
            sub_B4DE80((__int64)v33, (__int64)v24, (__int64)v26, (__int64)v69, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
            *(_QWORD *)(a2 + 88),
            v11,
            v66,
            *(_QWORD *)(a2 + 56),
            *(_QWORD *)(a2 + 64));
          v34 = *(_QWORD *)a2;
          v35 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
          if ( *(_QWORD *)a2 != v35 )
          {
            do
            {
              v36 = *(_QWORD *)(v34 + 8);
              v37 = *(_DWORD *)v34;
              v34 += 16;
              sub_B99FD0(v11, v37, v36);
            }
            while ( v35 != v34 );
          }
          return v11;
        }
        v11 = sub_AD5840((__int64)v24, v26, 0);
      }
      else
      {
        v11 = v32(v31, v24, v26, v30, v56);
      }
      if ( v11 )
        return v11;
      goto LABEL_21;
    }
  }
  if ( *a1 )
  {
    v69[0].m128i_i64[0] = sub_9208B0((__int64)a1, v62);
    v69[0].m128i_i64[1] = v44;
    v59 = v69[0].m128i_i64[0] + 7;
    a4 = sub_9208B0((__int64)a1, (__int64)v8);
    v69[0].m128i_i64[1] = v45;
    v69[0].m128i_i64[0] = a4;
    v10 = 8 * ((v59 >> 3) - ((unsigned __int64)(a4 + 7) >> 3) - v7);
  }
  else
  {
    v10 = 8 * v7;
  }
  if ( v10 )
  {
    v65 = 1;
    v63[0].m128i_i64[0] = (__int64)".shift";
    v64 = 3;
    sub_9C6370(v66, a6, v63, a4, a5, (__int64)a6);
    v38 = sub_AD64C0(*(_QWORD *)(v6 + 8), v10, 0);
    v39 = *(_QWORD *)(a2 + 80);
    v40 = (unsigned __int8 *)v38;
    v41 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8))(*(_QWORD *)v39 + 24LL);
    if ( v41 == sub_920250 )
    {
      if ( *(_BYTE *)v6 > 0x15u || *v40 > 0x15u )
        goto LABEL_36;
      v58 = v40;
      if ( (unsigned __int8)sub_AC47B0(26) )
        v42 = sub_AD5570(26, v6, v58, 0, 0);
      else
        v42 = sub_AABE40(0x1Au, (unsigned __int8 *)v6, v58);
      v40 = v58;
      v43 = v42;
    }
    else
    {
      v60 = v40;
      v55 = v41(v39, 26u, (_BYTE *)v6, v40, 0);
      v40 = v60;
      v43 = v55;
    }
    if ( v43 )
    {
LABEL_33:
      v6 = v43;
      goto LABEL_5;
    }
LABEL_36:
    v70 = 257;
    v43 = sub_B504D0(26, v6, (__int64)v40, (__int64)v69, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v43,
      v66,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v46 = *(_QWORD *)a2;
    v47 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v47 )
    {
      do
      {
        v48 = *(_QWORD *)(v46 + 8);
        v49 = *(_DWORD *)v46;
        v46 += 16;
        sub_B99FD0(v43, v49, v48);
      }
      while ( v47 != v46 );
    }
    goto LABEL_33;
  }
LABEL_5:
  v11 = v6;
  if ( (__int64 *)v62 == v8 )
    return v11;
  v65 = 1;
  v63[0].m128i_i64[0] = (__int64)".trunc";
  v64 = 3;
  sub_9C6370(v66, a6, v63, a4, a5, (__int64)a6);
  if ( v8 == *(__int64 **)(v6 + 8) )
    return v11;
  v12 = *(_QWORD *)(a2 + 80);
  v13 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v12 + 120LL);
  if ( v13 != sub_920130 )
  {
    v11 = v13(v12, 38u, (_BYTE *)v6, (__int64)v8);
    goto LABEL_11;
  }
  if ( *(_BYTE *)v6 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v11 = sub_ADAB70(38, v6, (__int64 **)v8, 0);
    else
      v11 = sub_AA93C0(0x26u, v6, (__int64)v8);
LABEL_11:
    if ( v11 )
      return v11;
  }
  v70 = 257;
  v11 = sub_B51D30(38, v6, (__int64)v8, (__int64)v69, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v11,
    v66,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v50 = 16LL * *(unsigned int *)(a2 + 8);
  v51 = *(_QWORD *)a2;
  v52 = v51 + v50;
  while ( v52 != v51 )
  {
    v53 = *(_QWORD *)(v51 + 8);
    v54 = *(_DWORD *)v51;
    v51 += 16;
    sub_B99FD0(v11, v54, v53);
  }
  return v11;
}
