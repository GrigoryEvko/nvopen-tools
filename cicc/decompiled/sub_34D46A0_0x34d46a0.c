// Function: sub_34D46A0
// Address: 0x34d46a0
//
unsigned __int64 __fastcall sub_34D46A0(
        __int64 a1,
        int a2,
        _QWORD **a3,
        unsigned __int8 a4,
        char a5,
        char a6,
        int a7,
        unsigned int a8)
{
  __int64 v8; // r13
  __int64 v12; // rcx
  signed __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // r9
  unsigned __int64 v19; // rax
  bool v20; // of
  unsigned __int64 v21; // rax
  unsigned __int64 result; // rax
  int v23; // edx
  unsigned int v24; // esi
  unsigned __int64 v25; // rax
  int v26; // r12d
  unsigned int v27; // r14d
  __int64 *v28; // rsi
  unsigned int v29; // eax
  unsigned __int64 v30; // rax
  unsigned int v31; // eax
  __int64 *v32; // rax
  __int64 v33; // rax
  __int64 v34; // r8
  unsigned int v35; // esi
  unsigned __int64 v36; // rax
  unsigned int v37; // r12d
  int v38; // r14d
  __int64 v39; // r13
  unsigned __int64 v40; // rbx
  unsigned __int64 v41; // rax
  __int64 *v42; // rsi
  unsigned int v43; // eax
  unsigned int v44; // esi
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rdx
  unsigned int v47; // r15d
  __int64 v48; // r14
  unsigned __int64 v49; // r13
  int v50; // r12d
  __int64 v51; // rbx
  unsigned __int64 v52; // rax
  __int64 *v53; // rsi
  unsigned int v54; // eax
  __int64 v55; // rdx
  unsigned __int64 v56; // rax
  __int64 v57; // [rsp+8h] [rbp-78h]
  unsigned __int8 v58; // [rsp+10h] [rbp-70h]
  int v59; // [rsp+18h] [rbp-68h]
  unsigned __int64 v60; // [rsp+20h] [rbp-60h]
  __int64 v62; // [rsp+30h] [rbp-50h]
  int v63; // [rsp+30h] [rbp-50h]
  __int64 v64; // [rsp+30h] [rbp-50h]
  unsigned int v65; // [rsp+38h] [rbp-48h]
  __int64 v66; // [rsp+38h] [rbp-48h]
  __int64 v67; // [rsp+38h] [rbp-48h]
  unsigned __int64 v68; // [rsp+38h] [rbp-48h]
  unsigned __int64 v69; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v70; // [rsp+48h] [rbp-38h]

  if ( *((_BYTE *)a3 + 8) == 18 )
    return 0;
  v60 = 0;
  v8 = a1;
  v65 = *((_DWORD *)a3 + 8);
  if ( a6 )
  {
    v32 = (__int64 *)sub_BCE3C0(*a3, 0);
    v33 = sub_BCDA70(v32, v65);
    v34 = v33;
    if ( *(_BYTE *)(v33 + 8) != 18 )
    {
      v35 = *(_DWORD *)(v33 + 32);
      v70 = v35;
      if ( v35 > 0x40 )
      {
        v64 = v33;
        sub_C43690((__int64)&v69, -1, 1);
        v34 = v64;
        v35 = v70;
        if ( *(_BYTE *)(v64 + 8) == 18 )
        {
LABEL_44:
          if ( v35 > 0x40 && v69 )
            j_j___libc_free_0_0(v69);
          goto LABEL_3;
        }
LABEL_34:
        if ( *(int *)(v34 + 32) <= 0 )
        {
          v60 = 0;
        }
        else
        {
          v63 = a2;
          v37 = 0;
          v38 = *(_DWORD *)(v34 + 32);
          v39 = v34;
          v58 = a4;
          v40 = 0;
          do
          {
            v41 = v69;
            if ( v35 > 0x40 )
              v41 = *(_QWORD *)(v69 + 8LL * (v37 >> 6));
            if ( (v41 & (1LL << v37)) != 0 )
            {
              v42 = (__int64 *)v39;
              if ( (unsigned int)*(unsigned __int8 *)(v39 + 8) - 17 <= 1 )
                v42 = **(__int64 ***)(v39 + 16);
              v43 = sub_34D06B0(a1, v42);
              v35 = v70;
              v20 = __OFADD__(v43, v40);
              v40 += v43;
              if ( v20 )
              {
                v40 = 0x8000000000000000LL;
                if ( v43 )
                  v40 = 0x7FFFFFFFFFFFFFFFLL;
              }
            }
            ++v37;
          }
          while ( v38 != v37 );
          v60 = v40;
          v8 = a1;
          a2 = v63;
          a4 = v58;
        }
        goto LABEL_44;
      }
      v36 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v35;
      if ( !v35 )
        v36 = 0;
      v69 = v36;
      if ( *(_BYTE *)(v34 + 8) != 18 )
        goto LABEL_34;
    }
  }
LABEL_3:
  v12 = a4;
  BYTE1(v12) = 1;
  v13 = sub_34D2F80(v8, a2, (__int64)a3[3], v12, a8, a7);
  if ( is_mul_ok(v13, v65) )
  {
    v62 = v13 * v65;
  }
  else if ( v13 <= 0 || (v62 = 0x7FFFFFFFFFFFFFFFLL, !v65) )
  {
    v62 = 0x8000000000000000LL;
  }
  if ( *((_BYTE *)a3 + 8) == 18 )
  {
    v14 = 0;
    goto LABEL_7;
  }
  v23 = *((_DWORD *)a3 + 8);
  v70 = v23;
  if ( (unsigned int)v23 > 0x40 )
  {
    sub_C43690((__int64)&v69, -1, 1);
    if ( *((_BYTE *)a3 + 8) == 18 )
    {
      v24 = v70;
      v14 = 0;
      goto LABEL_47;
    }
    v23 = *((_DWORD *)a3 + 8);
    v24 = v70;
  }
  else
  {
    v24 = v23;
    v25 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v23;
    if ( !v23 )
      v25 = 0;
    v69 = v25;
  }
  v14 = 0;
  if ( v23 > 0 )
  {
    v26 = v23;
    v59 = a2;
    v27 = 0;
    while ( 1 )
    {
      v30 = v69;
      if ( v24 > 0x40 )
        v30 = *(_QWORD *)(v69 + 8LL * (v27 >> 6));
      if ( (v30 & (1LL << v27)) == 0 )
        goto LABEL_22;
      v31 = *((unsigned __int8 *)a3 + 8) - 17;
      if ( v59 == 33 )
        break;
      v28 = (__int64 *)a3;
      if ( v31 <= 1 )
        goto LABEL_20;
LABEL_21:
      v29 = sub_34D06B0(v8, v28);
      v24 = v70;
      v20 = __OFADD__(v29, v14);
      v14 += v29;
      if ( v20 )
      {
        v14 = 0x8000000000000000LL;
        if ( v29 )
          v14 = 0x7FFFFFFFFFFFFFFFLL;
      }
LABEL_22:
      if ( v26 == ++v27 )
        goto LABEL_47;
    }
    v28 = (__int64 *)a3;
    if ( v31 > 1 )
      goto LABEL_21;
LABEL_20:
    v28 = (__int64 *)*a3[2];
    goto LABEL_21;
  }
LABEL_47:
  if ( v24 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
LABEL_7:
  if ( !a5 )
  {
    v15 = 0;
    goto LABEL_9;
  }
  v15 = ((a7 == 0) + 1LL) * v65;
  v16 = (__int64 *)sub_BCB2A0(*a3);
  v17 = sub_BCDA70(v16, v65);
  v18 = v17;
  if ( *(_BYTE *)(v17 + 8) == 18 )
    goto LABEL_9;
  v44 = *(_DWORD *)(v17 + 32);
  v70 = v44;
  if ( v44 > 0x40 )
  {
    v67 = v17;
    sub_C43690((__int64)&v69, -1, 1);
    v18 = v67;
    v44 = v70;
    if ( *(_BYTE *)(v67 + 8) != 18 )
      goto LABEL_58;
    if ( v70 > 0x40 )
    {
      v46 = 0;
      goto LABEL_77;
    }
  }
  else
  {
    v45 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v44;
    if ( !v44 )
      v45 = 0;
    v69 = v45;
    if ( *(_BYTE *)(v18 + 8) != 18 )
    {
LABEL_58:
      v46 = 0;
      if ( *(int *)(v18 + 32) > 0 )
      {
        v66 = v15;
        v47 = 0;
        v48 = v8;
        v49 = 0;
        v50 = *(_DWORD *)(v18 + 32);
        v57 = v14;
        v51 = v18;
        do
        {
          v52 = v69;
          if ( v44 > 0x40 )
            v52 = *(_QWORD *)(v69 + 8LL * (v47 >> 6));
          if ( (v52 & (1LL << v47)) != 0 )
          {
            v53 = (__int64 *)v51;
            if ( (unsigned int)*(unsigned __int8 *)(v51 + 8) - 17 <= 1 )
              v53 = **(__int64 ***)(v51 + 16);
            v54 = sub_34D06B0(v48, v53);
            v44 = v70;
            v20 = __OFADD__(v54, v49);
            v49 += v54;
            if ( v20 )
            {
              v49 = 0x8000000000000000LL;
              if ( v54 )
                v49 = 0x7FFFFFFFFFFFFFFFLL;
            }
          }
          ++v47;
        }
        while ( v50 != v47 );
        v15 = v66;
        v46 = v49;
        v14 = v57;
      }
      if ( v44 <= 0x40 )
      {
LABEL_69:
        v20 = __OFADD__(v15, v46);
        v55 = v15 + v46;
        if ( v20 )
        {
          v56 = 0x7FFFFFFFFFFFFFFFLL;
          if ( !v15 )
            v56 = 0x8000000000000000LL;
          v15 = v56;
          goto LABEL_9;
        }
        goto LABEL_101;
      }
LABEL_77:
      if ( v69 )
      {
        v68 = v46;
        j_j___libc_free_0_0(v69);
        v46 = v68;
      }
      goto LABEL_69;
    }
  }
  v55 = v15;
LABEL_101:
  v15 = v55;
LABEL_9:
  v19 = v62 + v60;
  if ( __OFADD__(v62, v60) )
  {
    v19 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v62 <= 0 )
      v19 = 0x8000000000000000LL;
  }
  v20 = __OFADD__(v14, v19);
  v21 = v14 + v19;
  if ( v20 )
  {
    v21 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v14 <= 0 )
      v21 = 0x8000000000000000LL;
  }
  v20 = __OFADD__(v15, v21);
  result = v15 + v21;
  if ( v20 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v15 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
