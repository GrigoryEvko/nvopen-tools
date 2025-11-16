// Function: sub_1165C10
// Address: 0x1165c10
//
__int64 __fastcall sub_1165C10(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 result; // rax
  unsigned __int8 v10; // al
  __int64 v11; // rdi
  __m128i v12; // xmm1
  unsigned __int64 v13; // xmm2_8
  __int64 v14; // rax
  __m128i v15; // xmm3
  char v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 v22; // r15
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r13
  __int64 v28; // r12
  _BYTE *v29; // rax
  bool v30; // al
  __int64 v31; // rbx
  __int64 v32; // r15
  __int64 v33; // r12
  __int64 v34; // rbx
  __int64 i; // r12
  __int64 v36; // rdx
  unsigned int v37; // esi
  bool v38; // dl
  unsigned int v39; // ecx
  __int64 v40; // rax
  unsigned int v41; // ecx
  int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // r12
  __int64 v45; // rbx
  __int64 v46; // r12
  __int64 v47; // rdx
  unsigned int v48; // esi
  int v49; // [rsp-D0h] [rbp-D0h]
  bool v50; // [rsp-CCh] [rbp-CCh]
  int v51; // [rsp-CCh] [rbp-CCh]
  int v52; // [rsp-C8h] [rbp-C8h]
  unsigned int v53; // [rsp-C8h] [rbp-C8h]
  __int64 v54; // [rsp-C8h] [rbp-C8h]
  _BYTE *v55; // [rsp-C0h] [rbp-C0h]
  unsigned __int8 *v56; // [rsp-C0h] [rbp-C0h]
  __int64 v57; // [rsp-C0h] [rbp-C0h]
  __int64 v58; // [rsp-C0h] [rbp-C0h]
  _BYTE v59[32]; // [rsp-B8h] [rbp-B8h] BYREF
  __int16 v60; // [rsp-98h] [rbp-98h]
  _OWORD v61[2]; // [rsp-88h] [rbp-88h] BYREF
  unsigned __int64 v62; // [rsp-68h] [rbp-68h]
  __int64 v63; // [rsp-60h] [rbp-60h]
  __m128i v64; // [rsp-58h] [rbp-58h]
  __int64 v65; // [rsp-48h] [rbp-48h]

  v5 = *(_QWORD *)(a1 + 16);
  if ( !v5 || *(_QWORD *)(v5 + 8) )
    return 0;
  v10 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 != 55 )
    goto LABEL_5;
  v11 = *(_QWORD *)(a1 - 64);
  v25 = *(_QWORD *)(v11 + 16);
  if ( !v25 )
    goto LABEL_9;
  if ( *(_QWORD *)(v25 + 8) )
    goto LABEL_9;
  if ( *(_BYTE *)v11 != 54 )
    goto LABEL_9;
  v26 = *(_QWORD *)(v11 - 64);
  v56 = (unsigned __int8 *)v26;
  if ( !v26 )
    goto LABEL_9;
  v27 = *(_QWORD *)(v11 - 32);
  if ( !v27 )
    goto LABEL_9;
  v28 = *(_QWORD *)(a1 - 32);
  if ( !v28 )
    goto LABEL_9;
  if ( *(_BYTE *)v26 == 17 )
  {
    if ( *(_DWORD *)(v26 + 32) > 0x40u )
    {
      v52 = *(_DWORD *)(v26 + 32);
      v29 = *(_BYTE **)(v11 - 64);
LABEL_33:
      v30 = v52 - 1 == (unsigned int)sub_C444A0((__int64)(v29 + 24));
      goto LABEL_34;
    }
    v30 = *(_QWORD *)(v26 + 24) == 1;
  }
  else
  {
    v54 = *(_QWORD *)(v26 + 8);
    v43 = (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17;
    if ( (unsigned int)v43 > 1 || *(_BYTE *)v26 > 0x15u )
      goto LABEL_9;
    v29 = sub_AD7630(v26, 0, v43);
    if ( !v29 || *v29 != 17 )
    {
      if ( *(_BYTE *)(v54 + 8) == 17 )
      {
        v49 = *(_DWORD *)(v54 + 32);
        if ( v49 )
        {
          v38 = 0;
          v39 = 0;
          while ( 1 )
          {
            v50 = v38;
            v53 = v39;
            v40 = sub_AD69F0(v56, v39);
            v41 = v53;
            v38 = v50;
            if ( !v40 )
              break;
            if ( *(_BYTE *)v40 != 13 )
            {
              if ( *(_BYTE *)v40 != 17 )
                break;
              if ( *(_DWORD *)(v40 + 32) <= 0x40u )
              {
                v38 = *(_QWORD *)(v40 + 24) == 1;
              }
              else
              {
                v51 = *(_DWORD *)(v40 + 32);
                v42 = sub_C444A0(v40 + 24);
                v41 = v53;
                v38 = v51 - 1 == v42;
              }
              if ( !v38 )
                break;
            }
            v39 = v41 + 1;
            if ( v49 == v39 )
            {
              if ( v38 )
                goto LABEL_35;
              goto LABEL_51;
            }
          }
        }
      }
      goto LABEL_51;
    }
    if ( *((_DWORD *)v29 + 8) > 0x40u )
    {
      v52 = *((_DWORD *)v29 + 8);
      goto LABEL_33;
    }
    v30 = *((_QWORD *)v29 + 3) == 1;
  }
LABEL_34:
  if ( !v30 )
  {
LABEL_51:
    v10 = *(_BYTE *)a1;
LABEL_5:
    if ( (unsigned __int8)(v10 - 54) > 1u )
      return 0;
    v11 = *(_QWORD *)(a1 - 64);
LABEL_9:
    v12 = _mm_loadu_si128(a2 + 7);
    v13 = _mm_loadu_si128(a2 + 8).m128i_u64[0];
    v14 = a2[10].m128i_i64[0];
    v15 = _mm_loadu_si128(a2 + 9);
    v61[0] = _mm_loadu_si128(a2 + 6);
    v62 = v13;
    v65 = v14;
    v63 = a3;
    v61[1] = v12;
    v64 = v15;
    v16 = sub_9A1DB0((unsigned __int8 *)v11, 0, 0, (__int64)v61, a5);
    if ( !v16 )
      return 0;
    v17 = sub_1165C10(*(_QWORD *)(a1 - 64), a2, a3);
    if ( v17 )
    {
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        v18 = *(_QWORD *)(a1 - 8);
      else
        v18 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v19 = *(_BYTE **)v18;
      if ( *(_QWORD *)v18 )
      {
        v20 = *(_QWORD *)(v18 + 8);
        **(_QWORD **)(v18 + 16) = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = *(_QWORD *)(v18 + 16);
      }
      *(_QWORD *)v18 = v17;
      v21 = *(_QWORD *)(v17 + 16);
      *(_QWORD *)(v18 + 8) = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = v18 + 8;
      *(_QWORD *)(v18 + 16) = v17 + 16;
      *(_QWORD *)(v17 + 16) = v18;
      if ( *v19 > 0x1Cu )
      {
        v22 = a2[2].m128i_i64[1];
        *(_QWORD *)&v61[0] = v19;
        v55 = v19;
        v23 = v22 + 2096;
        sub_11604F0(v23, (__int64 *)v61);
        v24 = *((_QWORD *)v55 + 2);
        if ( v24 )
        {
          if ( !*(_QWORD *)(v24 + 8) )
          {
            *(_QWORD *)&v61[0] = *(_QWORD *)(v24 + 24);
            sub_11604F0(v23, (__int64 *)v61);
          }
        }
      }
    }
    else
    {
      v16 = 0;
    }
    if ( *(_BYTE *)a1 == 55 )
    {
      if ( !sub_B44E60(a1) )
      {
        sub_B448B0(a1, 1);
        if ( *(_BYTE *)a1 != 54 || sub_B448F0(a1) )
          return a1;
        goto LABEL_66;
      }
    }
    else if ( *(_BYTE *)a1 == 54 && !sub_B448F0(a1) )
    {
LABEL_66:
      sub_B447F0((unsigned __int8 *)a1, 1);
      return a1;
    }
    if ( !v16 )
      return 0;
    return a1;
  }
LABEL_35:
  v31 = a2[2].m128i_i64[0];
  v60 = 257;
  v32 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v31 + 80) + 32LL))(
          *(_QWORD *)(v31 + 80),
          15,
          v27,
          v28,
          0,
          0);
  if ( !v32 )
  {
    LOWORD(v62) = 257;
    v32 = sub_B504D0(15, v27, v28, (__int64)v61, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v31 + 88) + 16LL))(
      *(_QWORD *)(v31 + 88),
      v32,
      v59,
      *(_QWORD *)(v31 + 56),
      *(_QWORD *)(v31 + 64));
    v44 = 16LL * *(unsigned int *)(v31 + 8);
    v45 = *(_QWORD *)v31;
    v46 = v45 + v44;
    while ( v46 != v45 )
    {
      v47 = *(_QWORD *)(v45 + 8);
      v48 = *(_DWORD *)v45;
      v45 += 16;
      sub_B99FD0(v32, v48, v47);
    }
  }
  v33 = a2[2].m128i_i64[0];
  v60 = 257;
  result = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned __int8 *, __int64, _QWORD, _QWORD))(**(_QWORD **)(v33 + 80)
                                                                                                  + 32LL))(
             *(_QWORD *)(v33 + 80),
             25,
             v56,
             v32,
             0,
             0);
  if ( !result )
  {
    LOWORD(v62) = 257;
    v57 = sub_B504D0(25, (__int64)v56, v32, (__int64)v61, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v33 + 88) + 16LL))(
      *(_QWORD *)(v33 + 88),
      v57,
      v59,
      *(_QWORD *)(v33 + 56),
      *(_QWORD *)(v33 + 64));
    v34 = *(_QWORD *)v33;
    result = v57;
    for ( i = *(_QWORD *)v33 + 16LL * *(unsigned int *)(v33 + 8); i != v34; result = v58 )
    {
      v36 = *(_QWORD *)(v34 + 8);
      v37 = *(_DWORD *)v34;
      v34 += 16;
      v58 = result;
      sub_B99FD0(result, v37, v36);
    }
  }
  return result;
}
