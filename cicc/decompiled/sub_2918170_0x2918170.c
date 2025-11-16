// Function: sub_2918170
// Address: 0x2918170
//
__int64 __fastcall sub_2918170(__int64 a1, __int64 a2, __int64 a3, __int64 a4, const __m128i *a5, __int64 a6)
{
  __int64 v7; // r12
  unsigned int v8; // ebx
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // r13
  __int64 result; // rax
  unsigned __int64 v13; // r8
  int v14; // edx
  unsigned int v15; // r15d
  unsigned int v16; // r12d
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdi
  _BYTE *v20; // r10
  __int64 (__fastcall *v21)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v22; // rax
  __int64 *v23; // rcx
  __int64 v24; // r8
  unsigned __int64 v25; // r9
  __int64 v26; // r15
  unsigned __int64 v27; // rdx
  int v28; // eax
  __int64 v29; // r15
  unsigned int v30; // r12d
  unsigned int v31; // r14d
  unsigned int v32; // ebx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  unsigned __int8 *v40; // r13
  __int64 (__fastcall *v41)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v42; // rax
  __int64 v43; // r9
  unsigned int *v44; // rbx
  __int64 v45; // r12
  __int64 v46; // rdx
  unsigned int v47; // esi
  _QWORD *v48; // rax
  unsigned int *v49; // rbx
  __int64 v50; // r14
  __int64 v51; // rdx
  unsigned int v52; // esi
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // [rsp+0h] [rbp-160h]
  __int64 v56; // [rsp+0h] [rbp-160h]
  _BYTE *v57; // [rsp+10h] [rbp-150h]
  __int64 v58; // [rsp+10h] [rbp-150h]
  _BYTE *v59; // [rsp+10h] [rbp-150h]
  unsigned int v60; // [rsp+20h] [rbp-140h]
  __int64 v61; // [rsp+20h] [rbp-140h]
  _DWORD *v63; // [rsp+40h] [rbp-120h]
  unsigned int v64; // [rsp+40h] [rbp-120h]
  __int64 v66; // [rsp+48h] [rbp-118h]
  __int64 v67; // [rsp+48h] [rbp-118h]
  _QWORD *v68; // [rsp+48h] [rbp-118h]
  __int64 v69; // [rsp+48h] [rbp-118h]
  __m128i v70[2]; // [rsp+50h] [rbp-110h] BYREF
  char v71; // [rsp+70h] [rbp-F0h]
  char v72; // [rsp+71h] [rbp-EFh]
  __m128i v73[2]; // [rsp+80h] [rbp-E0h] BYREF
  char v74; // [rsp+A0h] [rbp-C0h]
  char v75; // [rsp+A1h] [rbp-BFh]
  __m128i v76; // [rsp+B0h] [rbp-B0h] BYREF
  _BYTE v77[32]; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 *v78; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v79; // [rsp+E8h] [rbp-78h]
  _BYTE v80[16]; // [rsp+F0h] [rbp-70h] BYREF
  __int16 v81; // [rsp+100h] [rbp-60h]

  v7 = a1;
  v8 = a4;
  v9 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
  {
    v10 = *(_DWORD *)(v9 + 32);
    v11 = *(_QWORD *)(a2 + 8);
    result = a3;
    if ( v10 == *(_DWORD *)(v11 + 32) )
      return result;
    v60 = a4 + v10;
    v76.m128i_i64[0] = (__int64)v77;
    v76.m128i_i64[1] = 0x800000000LL;
    v13 = *(unsigned int *)(v11 + 32);
    v14 = v13;
    if ( (unsigned int)v13 > 8 )
    {
      sub_C8D5F0((__int64)&v76, v77, v13, 4u, v13, a6);
      v14 = *(_DWORD *)(v11 + 32);
    }
    if ( v14 )
    {
      v15 = 0;
      do
      {
        while ( 1 )
        {
          v53 = v76.m128i_u32[2];
          a4 = v76.m128i_u32[3];
          v17 = v76.m128i_u32[2] + 1LL;
          if ( v8 <= v15 && v60 > v15 )
            break;
          if ( v76.m128i_u32[3] < v17 )
          {
            sub_C8D5F0((__int64)&v76, v77, v17, 4u, v13, a6);
            v53 = v76.m128i_u32[2];
          }
          ++v15;
          *(_DWORD *)(v76.m128i_i64[0] + 4 * v53) = -1;
          ++v76.m128i_i32[2];
          if ( *(_DWORD *)(v11 + 32) == v15 )
            goto LABEL_15;
        }
        v16 = v15 - v8;
        if ( v76.m128i_u32[3] < v17 )
        {
          sub_C8D5F0((__int64)&v76, v77, v17, 4u, v13, a6);
          v53 = v76.m128i_u32[2];
        }
        ++v15;
        *(_DWORD *)(v76.m128i_i64[0] + 4 * v53) = v16;
        ++v76.m128i_i32[2];
      }
      while ( *(_DWORD *)(v11 + 32) != v15 );
LABEL_15:
      v7 = a1;
    }
    v72 = 1;
    v70[0].m128i_i64[0] = (__int64)".expand";
    v71 = 3;
    sub_9C6370(v73, a5, v70, a4, v13, a6);
    v63 = (_DWORD *)v76.m128i_i64[0];
    v55 = v76.m128i_u32[2];
    v18 = sub_ACADE0(*(__int64 ***)(a3 + 8));
    v19 = *(_QWORD *)(v7 + 80);
    v20 = (_BYTE *)v18;
    v21 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v19 + 112LL);
    if ( v21 == sub_9B6630 )
    {
      if ( *(_BYTE *)a3 > 0x15u || *v20 > 0x15u )
        goto LABEL_45;
      v57 = v20;
      v22 = sub_AD5CE0(a3, (__int64)v20, v63, v55, 0);
      v20 = v57;
      v26 = v22;
    }
    else
    {
      v59 = v20;
      v54 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v21)(v19, a3, v20, v63, v55);
      v20 = v59;
      v26 = v54;
    }
    if ( v26 )
    {
LABEL_21:
      v78 = (__int64 *)v80;
      v79 = 0x800000000LL;
      v27 = *(unsigned int *)(v11 + 32);
      v28 = v27;
      if ( (unsigned int)v27 > 8 )
      {
        sub_C8D5F0((__int64)&v78, v80, v27, 8u, v24, v25);
        v28 = *(_DWORD *)(v11 + 32);
      }
      if ( v28 )
      {
        v56 = v26;
        v29 = v7;
        v30 = 0;
        v31 = v8;
        v32 = v60;
        do
        {
          v33 = sub_BCB2A0(*(_QWORD **)(v29 + 72));
          v34 = sub_ACD640(v33, (v31 <= v30) & (unsigned __int8)(v32 > v30), 0);
          v35 = (unsigned int)v79;
          v25 = (unsigned int)v79 + 1LL;
          if ( v25 > HIDWORD(v79) )
          {
            v61 = v34;
            sub_C8D5F0((__int64)&v78, v80, (unsigned int)v79 + 1LL, 8u, v24, v25);
            v35 = (unsigned int)v79;
            v34 = v61;
          }
          v23 = v78;
          ++v30;
          v78[v35] = v34;
          LODWORD(v79) = v79 + 1;
        }
        while ( *(_DWORD *)(v11 + 32) != v30 );
        v7 = v29;
        v26 = v56;
      }
      v72 = 1;
      v70[0].m128i_i64[0] = (__int64)"blend";
      v71 = 3;
      sub_9C6370(v73, a5, v70, (__int64)v23, v24, v25);
      v36 = sub_AD3730(v78, (unsigned int)v79);
      result = sub_B36550((unsigned int **)v7, v36, v26, a2, (__int64)v73, 0);
      if ( v78 != (__int64 *)v80 )
      {
        v66 = result;
        _libc_free((unsigned __int64)v78);
        result = v66;
      }
      if ( (_BYTE *)v76.m128i_i64[0] != v77 )
      {
        v67 = result;
        _libc_free(v76.m128i_u64[0]);
        return v67;
      }
      return result;
    }
LABEL_45:
    v58 = (__int64)v20;
    v81 = 257;
    v48 = sub_BD2C40(112, unk_3F1FE60);
    v26 = (__int64)v48;
    if ( v48 )
      sub_B4E9E0((__int64)v48, a3, v58, v63, v55, (__int64)&v78, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
      *(_QWORD *)(v7 + 88),
      v26,
      v73,
      *(_QWORD *)(v7 + 56),
      *(_QWORD *)(v7 + 64));
    v23 = (__int64 *)(*(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8));
    if ( *(__int64 **)v7 != v23 )
    {
      v64 = v8;
      v49 = *(unsigned int **)v7;
      v50 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
      do
      {
        v51 = *((_QWORD *)v49 + 1);
        v52 = *v49;
        v49 += 4;
        sub_B99FD0(v26, v52, v51);
      }
      while ( (unsigned int *)v50 != v49 );
      v8 = v64;
    }
    goto LABEL_21;
  }
  v75 = 1;
  v73[0].m128i_i64[0] = (__int64)".insert";
  v74 = 3;
  sub_9C6370(&v76, a5, v73, (__int64)".insert", (__int64)a5, a6);
  v37 = sub_BCB2D0(*(_QWORD **)(a1 + 72));
  v38 = sub_ACD640(v37, v8, 0);
  v39 = *(_QWORD *)(a1 + 80);
  v40 = (unsigned __int8 *)v38;
  v41 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v39 + 104LL);
  if ( v41 != sub_948040 )
  {
    result = v41(v39, (_BYTE *)a2, (_BYTE *)a3, v40);
LABEL_39:
    if ( result )
      return result;
    goto LABEL_40;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *(_BYTE *)a3 <= 0x15u && *v40 <= 0x15u )
  {
    result = sub_AD5A90(a2, (_BYTE *)a3, v40, 0);
    goto LABEL_39;
  }
LABEL_40:
  v81 = 257;
  v42 = sub_BD2C40(72, 3u);
  if ( v42 )
  {
    v68 = v42;
    sub_B4DFA0((__int64)v42, a2, a3, (__int64)v40, (__int64)&v78, v43, 0, 0);
    v42 = v68;
  }
  v69 = (__int64)v42;
  (*(void (__fastcall **)(_QWORD, _QWORD *, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
    *(_QWORD *)(v7 + 88),
    v42,
    &v76,
    *(_QWORD *)(v7 + 56),
    *(_QWORD *)(v7 + 64));
  v44 = *(unsigned int **)v7;
  result = v69;
  v45 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
  if ( v44 != (unsigned int *)v45 )
  {
    do
    {
      v46 = *((_QWORD *)v44 + 1);
      v47 = *v44;
      v44 += 4;
      sub_B99FD0(v69, v47, v46);
    }
    while ( (unsigned int *)v45 != v44 );
    return v69;
  }
  return result;
}
