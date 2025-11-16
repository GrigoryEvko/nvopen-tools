// Function: sub_31A6FE0
// Address: 0x31a6fe0
//
__int64 __fastcall sub_31A6FE0(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 v2; // rdi
  unsigned int v3; // r15d
  __int64 *v5; // rdi
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // r15d
  __int64 *v17; // r12
  int v18; // eax
  int v19; // ecx
  unsigned int v20; // r15d
  __int64 *v21; // rbx
  int v22; // r12d
  __int64 v23; // rdx
  unsigned __int64 v24; // rdi
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  _BYTE *v27; // rdi
  __int64 v28; // rax
  __int64 *v29; // rax
  __int64 v30; // rax
  __int64 *v31; // r15
  __int64 v32; // rbx
  __int64 v33; // r12
  __int64 v34; // r13
  unsigned __int8 *v35; // r14
  unsigned __int64 v36; // rax
  __int64 v37; // rcx
  unsigned int v38; // eax
  __int64 *v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rcx
  unsigned int v42; // eax
  __int64 *v43; // [rsp-20h] [rbp-140h]
  __int64 v44; // [rsp-18h] [rbp-138h]
  __int64 v45; // [rsp+10h] [rbp-110h]
  __int64 *v46; // [rsp+18h] [rbp-108h]
  int v47; // [rsp+20h] [rbp-100h]
  __int64 *v48; // [rsp+30h] [rbp-F0h]
  __int64 v49; // [rsp+30h] [rbp-F0h]
  unsigned __int8 v50; // [rsp+38h] [rbp-E8h]
  __int64 v51; // [rsp+38h] [rbp-E8h]
  __int64 *v52; // [rsp+48h] [rbp-D8h]
  __int64 v53; // [rsp+48h] [rbp-D8h]
  __int64 *v54; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+58h] [rbp-C8h]
  _BYTE v56[16]; // [rsp+60h] [rbp-C0h] BYREF
  _BYTE *v57; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v58; // [rsp+78h] [rbp-A8h]
  _BYTE v59[32]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 *v60; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v61; // [rsp+A8h] [rbp-78h]
  _BYTE v62[112]; // [rsp+B0h] [rbp-70h] BYREF

  v1 = (__int64 *)a1;
  v45 = sub_D47930(*(_QWORD *)a1);
  if ( !v45 )
  {
    v3 = 0;
    sub_2AB8760(
      (__int64)"Loop does not have a latch",
      26,
      "Cannot vectorize early exit loop",
      0x20u,
      (__int64)"NoLatchEarlyExit",
      16,
      *(__int64 **)(a1 + 64),
      *(_QWORD *)a1,
      0);
    return v3;
  }
  v2 = *(_QWORD *)a1;
  if ( *((_DWORD *)v1 + 30) || *((_DWORD *)v1 + 66) != *((_DWORD *)v1 + 65) )
  {
    v3 = 0;
    sub_2AB8760(
      (__int64)"Found reductions or recurrences in early-exit loop",
      50,
      "Cannot vectorize early exit loop with reductions or recurrences",
      0x3Fu,
      (__int64)"RecurrencesInEarlyExitLoop",
      26,
      (__int64 *)v1[8],
      v2,
      0);
    return v3;
  }
  v60 = (__int64 *)v62;
  v61 = 0x800000000LL;
  sub_D46D90(v2, (__int64)&v60);
  v5 = v60;
  v57 = v59;
  v58 = 0x400000000LL;
  v48 = &v60[(unsigned int)v61];
  if ( v48 == v60 )
  {
    v3 = 0;
    goto LABEL_50;
  }
  v52 = v60;
  v6 = 0;
  v7 = 0;
  v50 = 0;
  do
  {
    v8 = *v52;
    v9 = sub_DBAF70(*(_QWORD *)(v1[2] + 112), *v1, *v52, (__int64)&v57, 0);
    if ( sub_D96A50(v9) )
    {
      v12 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v12 == v8 + 48 )
      {
        v23 = *v1;
        v54 = (__int64 *)v56;
        v55 = 0x200000000LL;
LABEL_43:
        sub_2AB8760(
          (__int64)"Early exiting block does not have exactly two successors",
          56,
          "Incorrect number of successors from early exiting block",
          0x37u,
          (__int64)"EarlyExitTooManySuccessors",
          26,
          (__int64 *)v1[8],
          v23,
          0);
        goto LABEL_44;
      }
      if ( !v12 )
        BUG();
      v13 = v12 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v12 - 24) - 30 > 0xA )
      {
        HIDWORD(v55) = 2;
        v54 = (__int64 *)v56;
        v18 = 0;
        v47 = 0;
      }
      else
      {
        v16 = sub_B46E30(v13);
        v17 = (__int64 *)v56;
        v47 = v16;
        v55 = 0x200000000LL;
        v18 = 0;
        v54 = (__int64 *)v56;
        if ( (unsigned __int64)v16 > 2 )
        {
          sub_C8D5F0((__int64)&v54, v56, v16, 8u, v14, v15);
          v18 = v55;
          v17 = &v54[(unsigned int)v55];
        }
        v19 = v16;
        if ( v16 )
        {
          v46 = v1;
          v20 = 0;
          v21 = v17;
          v22 = v19;
          do
          {
            if ( v21 )
              *v21 = sub_B46EC0(v13, v20);
            ++v20;
            ++v21;
          }
          while ( v22 != v20 );
          v1 = v46;
          v18 = v55;
        }
      }
      v23 = *v1;
      LODWORD(v55) = v47 + v18;
      if ( v47 + v18 != 2 )
        goto LABEL_43;
      v24 = (unsigned __int64)v54;
      v6 = *v54;
      if ( *(_BYTE *)(v23 + 84) )
      {
        v25 = *(_QWORD **)(v23 + 64);
        v26 = &v25[*(unsigned int *)(v23 + 76)];
        if ( v25 == v26 )
          goto LABEL_27;
        while ( v6 != *v25 )
        {
          if ( v26 == ++v25 )
            goto LABEL_27;
        }
      }
      else
      {
        v29 = sub_C8CA60(v23 + 56, *v54);
        v24 = (unsigned __int64)v54;
        if ( !v29 )
        {
          v6 = *v54;
          goto LABEL_27;
        }
      }
      v6 = *(_QWORD *)(v24 + 8);
LABEL_27:
      if ( !v50 )
      {
        if ( (_BYTE *)v24 != v56 )
          _libc_free(v24);
        v50 = 1;
        v7 = v8;
        goto LABEL_31;
      }
      sub_2AB8760(
        (__int64)"Loop has too many uncountable exits",
        35,
        "Cannot vectorize early exit loop with more than one early exit",
        0x3Eu,
        (__int64)"TooManyUncountableEarlyExits",
        28,
        (__int64 *)v1[8],
        *v1,
        0);
LABEL_44:
      if ( v54 != (__int64 *)v56 )
        _libc_free((unsigned __int64)v54);
      v27 = v57;
      v3 = 0;
      goto LABEL_47;
    }
    v28 = *((unsigned int *)v1 + 152);
    if ( v28 + 1 > (unsigned __int64)*((unsigned int *)v1 + 153) )
    {
      sub_C8D5F0((__int64)(v1 + 75), v1 + 77, v28 + 1, 8u, v10, v11);
      v28 = *((unsigned int *)v1 + 152);
    }
    *(_QWORD *)(v1[75] + 8 * v28) = v8;
    ++*((_DWORD *)v1 + 152);
LABEL_31:
    ++v52;
  }
  while ( v48 != v52 );
  v3 = v50;
  LODWORD(v58) = 0;
  if ( !v50 )
    goto LABEL_55;
  if ( v7 != sub_AA5510(v45) )
  {
    v3 = 0;
    sub_2AB8760(
      (__int64)"Early exit is not the latch predecessor",
      39,
      "Cannot vectorize early exit loop",
      0x20u,
      (__int64)"EarlyExitNotLatchPredecessor",
      28,
      (__int64 *)v1[8],
      *v1,
      0);
    v27 = v57;
    goto LABEL_47;
  }
  v30 = sub_DBAF70(*(_QWORD *)(v1[2] + 112), *v1, v45, (__int64)&v57, 0);
  if ( sub_D96A50(v30) )
  {
    v3 = 0;
    sub_2AB8760(
      (__int64)"Cannot determine exact exit count for latch block",
      49,
      "Cannot vectorize early exit loop",
      0x20u,
      (__int64)"UnknownLatchExitCountEarlyExitLoop",
      34,
      (__int64 *)v1[8],
      *v1,
      0);
    v27 = v57;
    goto LABEL_47;
  }
  v53 = *(_QWORD *)(*v1 + 40);
  if ( *(_QWORD *)(*v1 + 32) == v53 )
  {
LABEL_72:
    v40 = v1[2];
    v41 = v1[54];
    LODWORD(v58) = 0;
    v3 = sub_D31FD0(*v1, *(_QWORD *)(v40 + 112), (__int64 *)v1[5], v41, (__int64)&v57);
    if ( !(_BYTE)v3 )
    {
      sub_2AB8760(
        (__int64)"Loop may fault",
        14,
        "Cannot vectorize potentially faulting early exit loop",
        0x35u,
        (__int64)"PotentiallyFaultingEarlyExitLoop",
        32,
        (__int64 *)v1[8],
        *v1,
        0);
      v27 = v57;
      goto LABEL_47;
    }
    sub_DEF9D0((_QWORD *)v1[2]);
    v42 = *((unsigned __int8 *)v1 + 664);
    v1[81] = v7;
    v1[82] = v6;
    if ( (_BYTE)v42 )
    {
      v27 = v57;
      v3 = v42;
      goto LABEL_47;
    }
    *((_BYTE *)v1 + 664) = 1;
LABEL_55:
    v27 = v57;
    goto LABEL_47;
  }
  v51 = v7;
  v31 = v1;
  v32 = *(_QWORD *)(*v1 + 32);
  v49 = v6;
  while ( 1 )
  {
    v33 = *(_QWORD *)(*(_QWORD *)v32 + 56LL);
    v34 = *(_QWORD *)v32 + 48LL;
    if ( v34 != v33 )
      break;
LABEL_70:
    v32 += 8;
    if ( v53 == v32 )
    {
      v7 = v51;
      v6 = v49;
      v1 = v31;
      goto LABEL_72;
    }
  }
  while ( 1 )
  {
    v35 = (unsigned __int8 *)(v33 - 24);
    if ( !v33 )
      v35 = 0;
    if ( (unsigned __int8)sub_B46490((__int64)v35) )
      break;
    v36 = (unsigned int)*v35 - 29;
    if ( (unsigned int)v36 > 0x37 || (v37 = 0x80000300000004LL, !_bittest64(&v37, v36)) )
    {
      LOBYTE(v38) = sub_991A70(v35, 0, 0, 0, 0, 1u, 0);
      if ( !(_BYTE)v38 )
      {
        v39 = v31;
        v3 = v38;
        sub_2AB8760(
          (__int64)"Early exit loop contains operations that cannot be speculatively executed",
          73,
          "Early exit loop contains operations that cannot be speculatively executed",
          0x49u,
          (__int64)"UnsafeOperationsEarlyExitLoop",
          29,
          (__int64 *)v39[8],
          *v39,
          0);
        v27 = v57;
        goto LABEL_47;
      }
    }
    v33 = *(_QWORD *)(v33 + 8);
    if ( v34 == v33 )
      goto LABEL_70;
  }
  v44 = *v31;
  v43 = (__int64 *)v31[8];
  v3 = 0;
  sub_2AB8760(
    (__int64)"Writes to memory unsupported in early exit loops",
    48,
    "Cannot vectorize early exit loop with writes to memory",
    0x36u,
    (__int64)"WritesInEarlyExitLoop",
    21,
    v43,
    v44,
    0);
  v27 = v57;
LABEL_47:
  if ( v27 != v59 )
    _libc_free((unsigned __int64)v27);
  v5 = v60;
LABEL_50:
  if ( v5 != (__int64 *)v62 )
    _libc_free((unsigned __int64)v5);
  return v3;
}
