// Function: sub_1B78950
// Address: 0x1b78950
//
void __fastcall sub_1B78950(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        int a14)
{
  __int64 v16; // r14
  __int64 *v17; // r13
  __int64 *v18; // r14
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r13
  __int64 v24; // r14
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  bool v28; // zf
  __int64 v29; // r14
  _BYTE *v30; // r13
  __int64 v31; // r14
  double v32; // xmm4_8
  double v33; // xmm5_8
  __int64 v34; // rdi
  unsigned __int8 v35; // al
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rax
  _BYTE *v40; // rdi
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v43; // rdx
  _QWORD *v44; // r14
  _QWORD *v45; // r15
  int v46; // r8d
  int v47; // r9d
  __int64 v48; // r13
  __int64 v49; // rax
  unsigned int v50; // edx
  __int64 v51; // r12
  _QWORD *v52; // r14
  unsigned __int8 v53; // r15
  __int64 *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  _BYTE *v57; // rdi
  _QWORD *v58; // [rsp+18h] [rbp-D8h]
  _BYTE *v59; // [rsp+28h] [rbp-C8h]
  __int64 v60; // [rsp+28h] [rbp-C8h]
  __int64 v61; // [rsp+30h] [rbp-C0h] BYREF
  char v62; // [rsp+38h] [rbp-B8h]
  _BYTE *v63; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v64; // [rsp+48h] [rbp-A8h]
  _BYTE v65[32]; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE *v66; // [rsp+70h] [rbp-80h] BYREF
  __int64 v67; // [rsp+78h] [rbp-78h]
  _BYTE v68[112]; // [rsp+80h] [rbp-70h] BYREF

  v16 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v17 = *(__int64 **)(a2 - 8);
    v18 = &v17[v16];
  }
  else
  {
    v17 = (__int64 *)(a2 - v16 * 8);
    v18 = (__int64 *)a2;
  }
  for ( ; v18 != v17; v17 += 3 )
  {
    v19 = sub_1B75C50(a1, *v17, *(double *)a3.m128_u64, a4, a5);
    if ( v19 )
    {
      if ( *v17 )
      {
        v20 = v17[1];
        v21 = v17[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v21 = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = *(_QWORD *)(v20 + 16) & 3LL | v21;
      }
      *v17 = v19;
      v22 = *(_QWORD *)(v19 + 8);
      v17[1] = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = (unsigned __int64)(v17 + 1) | *(_QWORD *)(v22 + 16) & 3LL;
      v17[2] = v17[2] & 3 | (v19 + 8);
      *(_QWORD *)(v19 + 8) = v17;
    }
  }
  if ( *(_BYTE *)(a2 + 16) == 77 && (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
  {
    v23 = 0;
    v24 = 8LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    do
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v25 = *(_QWORD *)(a2 - 8);
      else
        v25 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v26 = sub_1B75C50(
              a1,
              *(_QWORD *)(v23 + v25 + 24LL * *(unsigned int *)(a2 + 56) + 8),
              *(double *)a3.m128_u64,
              a4,
              a5);
      if ( v26 )
      {
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v27 = *(_QWORD *)(a2 - 8);
        else
          v27 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        *(_QWORD *)(v23 + v27 + 24LL * *(unsigned int *)(a2 + 56) + 8) = v26;
      }
      v23 += 8;
    }
    while ( v24 != v23 );
  }
  v28 = *(_QWORD *)(a2 + 48) == 0;
  v66 = v68;
  v67 = 0x400000000LL;
  if ( v28 && *(__int16 *)(a2 + 18) >= 0 )
  {
    v34 = *(_QWORD *)(a1 + 8);
    if ( !v34 )
      return;
  }
  else
  {
    sub_161F840(a2, (__int64)&v66);
    v29 = 16LL * (unsigned int)v67;
    v59 = &v66[v29];
    if ( v66 != &v66[v29] )
    {
      v30 = v66;
      do
      {
        v31 = *((_QWORD *)v30 + 1);
        sub_1B76840((__int64)&v61, a1, v31, *(double *)a3.m128_u64, a4, a5);
        if ( v62 )
          a13 = v61;
        else
          a13 = sub_1B785E0(a1, v31, a3, a4, a5, a6, v32, v33, a9, a10);
        if ( v31 != a13 )
          sub_1625C10(a2, *(_DWORD *)v30, a13);
        v30 += 16;
      }
      while ( v59 != v30 );
    }
    v34 = *(_QWORD *)(a1 + 8);
    if ( !v34 )
      goto LABEL_55;
  }
  v35 = *(_BYTE *)(a2 + 16);
  if ( v35 <= 0x17u )
    goto LABEL_38;
  v36 = a2 | 4;
  if ( v35 == 78 || (v36 = a2 & 0xFFFFFFFFFFFFFFFBLL, v35 == 29) )
  {
    v58 = (_QWORD *)(v36 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v36 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_38;
    v63 = v65;
    v64 = 0x300000000LL;
    v60 = v58[8];
    v41 = *(unsigned int *)(v60 + 12);
    v42 = (unsigned int)(v41 - 1);
    if ( v42 > 3 )
    {
      sub_16CD150((__int64)&v63, v65, v42, 8, a13, a14);
      v34 = *(_QWORD *)(a1 + 8);
      v41 = *(unsigned int *)(v60 + 12);
    }
    v43 = *(_QWORD *)(v60 + 16);
    v44 = (_QWORD *)(v43 + 8 * v41);
    if ( v44 == (_QWORD *)(v43 + 8) )
    {
      v50 = v64;
    }
    else
    {
      v45 = (_QWORD *)(v43 + 8);
      do
      {
        v48 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v34 + 24LL))(v34, *v45);
        v49 = (unsigned int)v64;
        if ( (unsigned int)v64 >= HIDWORD(v64) )
        {
          sub_16CD150((__int64)&v63, v65, 0, 8, v46, v47);
          v49 = (unsigned int)v64;
        }
        ++v45;
        *(_QWORD *)&v63[8 * v49] = v48;
        v34 = *(_QWORD *)(a1 + 8);
        v50 = v64 + 1;
        LODWORD(v64) = v64 + 1;
      }
      while ( v44 != v45 );
    }
    v51 = v50;
    v52 = v63;
    v53 = *(_DWORD *)(v60 + 8) >> 8 != 0;
    v54 = (__int64 *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v34 + 24LL))(v34, *(_QWORD *)a2);
    v55 = sub_1644EA0(v54, v52, v51, v53);
    v56 = **(_QWORD **)(v55 + 16);
    v58[8] = v55;
    v57 = v63;
    *v58 = v56;
    if ( v57 != v65 )
      _libc_free((unsigned __int64)v57);
LABEL_55:
    v40 = v66;
    if ( v66 == v68 )
      return;
LABEL_41:
    _libc_free((unsigned __int64)v40);
    return;
  }
  if ( v35 == 53 )
  {
    *(_QWORD *)(a2 + 56) = (*(__int64 (__fastcall **)(__int64, _QWORD, unsigned __int64))(*(_QWORD *)v34 + 24LL))(
                             v34,
                             *(_QWORD *)(a2 + 56),
                             v36);
    v35 = *(_BYTE *)(a2 + 16);
    v34 = *(_QWORD *)(a1 + 8);
  }
LABEL_38:
  if ( v35 == 56 )
  {
    v37 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v34 + 24LL))(v34, *(_QWORD *)(a2 + 56));
    v38 = *(_QWORD *)(a2 + 64);
    *(_QWORD *)(a2 + 56) = v37;
    *(_QWORD *)(a2 + 64) = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(
                             *(_QWORD *)(a1 + 8),
                             v38);
    v34 = *(_QWORD *)(a1 + 8);
  }
  v39 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v34 + 24LL))(v34, *(_QWORD *)a2);
  v40 = v66;
  *(_QWORD *)a2 = v39;
  if ( v40 != v68 )
    goto LABEL_41;
}
