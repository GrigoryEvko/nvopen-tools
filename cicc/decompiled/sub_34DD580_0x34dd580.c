// Function: sub_34DD580
// Address: 0x34dd580
//
unsigned __int64 __fastcall sub_34DD580(
        __int64 a1,
        unsigned int a2,
        char a3,
        __int64 *a4,
        __int64 a5,
        int a6,
        unsigned int a7)
{
  char v7; // r15
  __int64 v8; // r14
  __int64 v9; // r13
  unsigned int v11; // ebx
  char v12; // al
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r10
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rbx
  __int64 v18; // rcx
  unsigned __int64 result; // rax
  __int64 v20; // rax
  unsigned int v21; // esi
  unsigned __int64 v22; // rax
  int v23; // r14d
  unsigned __int64 v24; // rbx
  unsigned int v25; // r15d
  __int64 v26; // r13
  unsigned __int64 v27; // rax
  __int64 *v28; // rsi
  unsigned int v29; // eax
  bool v30; // of
  signed __int64 v31; // rax
  int v32; // edx
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // r15
  __int64 v36; // r12
  __int64 v37; // rax
  __int128 v38; // [rsp-18h] [rbp-118h]
  __int64 v39; // [rsp+0h] [rbp-100h]
  char v40; // [rsp+8h] [rbp-F8h]
  unsigned int v41; // [rsp+Ch] [rbp-F4h]
  __int64 *v42; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v43; // [rsp+10h] [rbp-F0h]
  __int64 v44; // [rsp+18h] [rbp-E8h]
  __int64 v45; // [rsp+18h] [rbp-E8h]
  __int64 v46; // [rsp+18h] [rbp-E8h]
  __int64 v47; // [rsp+18h] [rbp-E8h]
  __int64 v48; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v49; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v50; // [rsp+18h] [rbp-E8h]
  __int64 v51; // [rsp+28h] [rbp-D8h] BYREF
  unsigned __int64 v52; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v53; // [rsp+38h] [rbp-C8h]
  char *v54; // [rsp+48h] [rbp-B8h]
  char v55; // [rsp+58h] [rbp-A8h] BYREF
  char *v56; // [rsp+78h] [rbp-88h]
  char v57; // [rsp+88h] [rbp-78h] BYREF

  v7 = a3;
  v8 = a1;
  v9 = a5;
  v11 = a2;
  v12 = *(_BYTE *)(a5 + 8);
  if ( v12 != 17 || a2 != 13 || !a3 )
  {
LABEL_4:
    v13 = *(_DWORD *)(v9 + 32);
    BYTE4(v51) = v12 == 18;
    LODWORD(v51) = v13;
    v14 = sub_BCE1B0(a4, v51);
    v15 = v14;
    if ( (a6 & 1) != 0 )
    {
      v44 = v14;
      v16 = sub_34D61B0(a1, a2, (_QWORD **)v14, a7);
      v15 = v44;
      v17 = v16;
      goto LABEL_6;
    }
    if ( *(_BYTE *)(v14 + 8) == 18 )
    {
      v17 = 0;
      goto LABEL_6;
    }
    v21 = *(_DWORD *)(v14 + 32);
    v53 = v21;
    if ( v21 > 0x40 )
    {
      v48 = v14;
      sub_C43690((__int64)&v52, -1, 1);
      v15 = v48;
      v21 = v53;
      if ( *(_BYTE *)(v48 + 8) == 18 )
      {
        v43 = 0;
LABEL_26:
        if ( v21 > 0x40 && v52 )
        {
          v46 = v15;
          j_j___libc_free_0_0(v52);
          v15 = v46;
        }
LABEL_29:
        v47 = v15;
        v31 = sub_34D2250(v8, v11, *(_QWORD *)(v15 + 24), a7, 0, 0, 0, 0, 0);
        v15 = v47;
        v33 = *(unsigned int *)(v47 + 32) * v31;
        if ( is_mul_ok(*(unsigned int *)(v47 + 32), v31) )
        {
          v34 = v33 + v43;
          if ( __OFADD__(v33, v43) )
          {
            v17 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v33 > 0 )
              goto LABEL_6;
            goto LABEL_32;
          }
          goto LABEL_39;
        }
        if ( *(_DWORD *)(v47 + 32) && v31 > 0 )
        {
          if ( v32 != 1 )
          {
            v17 = 0x7FFFFFFFFFFFFFFFLL;
            v34 = v43 + 0x7FFFFFFFFFFFFFFFLL;
            if ( __OFADD__(0x7FFFFFFFFFFFFFFFLL, v43) )
              goto LABEL_6;
            goto LABEL_39;
          }
          v34 = 0x7FFFFFFFFFFFFFFFLL;
          v17 = v43 + 0x7FFFFFFFFFFFFFFFLL;
          if ( __OFADD__(0x7FFFFFFFFFFFFFFFLL, v43) )
LABEL_39:
            v17 = v34;
LABEL_6:
          v18 = sub_34D3270(v8, (unsigned int)(v7 == 0) + 39, v15, v9, 0, a7, 0);
          result = v18 + v17;
          if ( __OFADD__(v18, v17) )
          {
            result = 0x7FFFFFFFFFFFFFFFLL;
            if ( v18 <= 0 )
              return 0x8000000000000000LL;
          }
          return result;
        }
        if ( v32 == 1 )
        {
          v17 = v43 + 0x8000000000000000LL;
          if ( !__OFADD__(v43, 0x8000000000000000LL) )
            goto LABEL_6;
        }
        else
        {
          v17 = v43 + 0x8000000000000000LL;
          if ( !__OFADD__(v43, 0x8000000000000000LL) )
            goto LABEL_6;
        }
LABEL_32:
        v17 = 0x8000000000000000LL;
        goto LABEL_6;
      }
    }
    else
    {
      v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
      if ( !v21 )
        v22 = 0;
      v52 = v22;
      if ( *(_BYTE *)(v15 + 8) == 18 )
      {
        v43 = 0;
        goto LABEL_29;
      }
    }
    if ( *(int *)(v15 + 32) <= 0 )
    {
      v43 = 0;
    }
    else
    {
      v41 = v11;
      v40 = v7;
      v23 = *(_DWORD *)(v15 + 32);
      v24 = 0;
      v39 = v9;
      v25 = 0;
      v26 = v15;
      do
      {
        v27 = v52;
        if ( v21 > 0x40 )
          v27 = *(_QWORD *)(v52 + 8LL * (v25 >> 6));
        if ( (v27 & (1LL << v25)) != 0 )
        {
          v28 = (__int64 *)v26;
          if ( (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17 <= 1 )
            v28 = **(__int64 ***)(v26 + 16);
          v29 = sub_34D06B0(a1, v28);
          v21 = v53;
          v30 = __OFADD__(v29, v24);
          v24 += v29;
          if ( v30 )
          {
            v24 = 0x8000000000000000LL;
            if ( v29 )
              v24 = 0x7FFFFFFFFFFFFFFFLL;
          }
        }
        ++v25;
      }
      while ( v23 != v25 );
      v43 = v24;
      v15 = v26;
      v11 = v41;
      v8 = a1;
      v7 = v40;
      v9 = v39;
    }
    goto LABEL_26;
  }
  v42 = a4;
  v45 = *(_QWORD *)(a5 + 24);
  v20 = sub_BCB2A0(*(_QWORD **)a5);
  a4 = v42;
  if ( v45 != v20 )
  {
    v12 = *(_BYTE *)(v9 + 8);
    goto LABEL_4;
  }
  v35 = sub_BCCE00((_QWORD *)*v42, *(_DWORD *)(v9 + 32));
  v51 = v35;
  *((_QWORD *)&v38 + 1) = 1;
  *(_QWORD *)&v38 = 0;
  sub_DF8CB0((__int64)&v52, 66, v35, (char *)&v51, 1, a6, 0, v38);
  v36 = sub_34D6FB0(a1, (__int64)&v52, a7);
  v37 = sub_34D3270(a1, 0x31u, v35, v9, 0, a7, 0);
  v30 = __OFADD__(v36, v37);
  result = v36 + v37;
  if ( v30 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v36 <= 0 )
      result = 0x8000000000000000LL;
  }
  if ( v56 != &v57 )
  {
    v49 = result;
    _libc_free((unsigned __int64)v56);
    result = v49;
  }
  if ( v54 != &v55 )
  {
    v50 = result;
    _libc_free((unsigned __int64)v54);
    return v50;
  }
  return result;
}
