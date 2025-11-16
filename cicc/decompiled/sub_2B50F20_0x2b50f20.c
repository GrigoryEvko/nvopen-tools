// Function: sub_2B50F20
// Address: 0x2b50f20
//
__int64 __fastcall sub_2B50F20(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  signed __int64 v5; // r13
  unsigned __int8 **v6; // r14
  __int64 result; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r11
  int v13; // r10d
  unsigned int v14; // r10d
  _BYTE *v15; // r14
  unsigned __int8 **v16; // r8
  unsigned __int8 **v17; // rcx
  char v18; // cl
  __int64 v19; // r11
  __int64 v20; // r8
  char *v21; // rdi
  unsigned int v22; // r10d
  unsigned __int8 **v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r13
  unsigned __int8 *v27; // rsi
  __int64 v28; // rdx
  unsigned __int8 **v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rdx
  unsigned __int8 **v32; // rax
  bool v33; // cl
  __int64 *v34; // r12
  __int64 v35; // r14
  __int64 v36; // rcx
  unsigned __int8 **v37; // rax
  unsigned __int64 v38; // r10
  unsigned __int8 **v39; // rcx
  __int64 *v40; // r12
  int v41; // eax
  int v42; // [rsp+10h] [rbp-E0h]
  __int64 **v43; // [rsp+18h] [rbp-D8h]
  __int64 v45; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v46; // [rsp+30h] [rbp-C0h]
  __int64 v48; // [rsp+38h] [rbp-B8h]
  __int64 v49; // [rsp+38h] [rbp-B8h]
  __int64 v50; // [rsp+38h] [rbp-B8h]
  void *s2; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+48h] [rbp-A8h]
  _BYTE v53[48]; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE *v54; // [rsp+80h] [rbp-70h] BYREF
  __int64 v55; // [rsp+88h] [rbp-68h]
  _BYTE v56[96]; // [rsp+90h] [rbp-60h] BYREF

  v5 = 8 * a3;
  v6 = (unsigned __int8 **)(a2 + 8 * a3);
  v46 = a3;
  if ( !a4 && v6 == sub_2B0BF30((_QWORD *)a2, (__int64)v6, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0)
    || v6 == sub_2B0C060((unsigned __int8 **)a2, (__int64)v6) )
  {
    return 0;
  }
  v43 = (__int64 **)sub_2B08680(*(_QWORD *)a1, a3);
  v12 = v5 >> 3;
  s2 = v53;
  v52 = 0x600000000LL;
  if ( (unsigned __int64)v5 > 0x30 )
  {
    sub_C8D5F0((__int64)&s2, v53, v5 >> 3, 8u, v10, v11);
    v12 = v5 >> 3;
    v21 = (char *)s2 + 8 * (unsigned int)v52;
  }
  else
  {
    v13 = 0;
    if ( !v5 )
      goto LABEL_7;
    v21 = v53;
  }
  v42 = v12;
  memcpy(v21, (const void *)a2, v5);
  v13 = v52;
  LODWORD(v12) = v42;
LABEL_7:
  v14 = v12 + v13;
  LODWORD(v52) = v14;
  if ( a4 )
  {
    v15 = s2;
    v46 = v14;
    v16 = sub_2B0B400((unsigned __int8 **)s2, (__int64)s2 + 8 * v14);
    result = 0;
    if ( v17 == v16 )
      goto LABEL_9;
    v18 = 0;
    v19 = *(_QWORD *)(a1 + 184);
    v20 = *(_QWORD *)a1;
    goto LABEL_12;
  }
  if ( sub_2B08550((unsigned __int8 **)a2, a3) )
  {
    v23 = sub_2B0C060((unsigned __int8 **)a2, (__int64)v6);
    v26 = (__int64)v23;
    if ( v6 == (unsigned __int8 **)a2 )
      goto LABEL_35;
    v27 = *v23;
    v28 = 0;
    v29 = (unsigned __int8 **)a2;
    do
    {
      v30 = *v29++ == v27;
      v28 += v30;
    }
    while ( v6 != v29 );
    if ( v28 <= 1 || *(unsigned __int8 **)a2 == v27 && v6 == sub_2B0C060((unsigned __int8 **)(a2 + 8), (__int64)v6) )
    {
LABEL_35:
      v40 = *(__int64 **)(a1 + 112);
      if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 17 )
      {
        result = sub_DFBC30(
                   v40,
                   4,
                   (__int64)v43,
                   0,
                   0,
                   0,
                   *(_DWORD *)(*(_QWORD *)a1 + 32LL) * (unsigned int)((v26 - a2) >> 3),
                   *(_QWORD *)a1,
                   0,
                   0,
                   0);
      }
      else
      {
        sub_ACADE0(v43);
        result = sub_DFD330(v40);
      }
    }
    else
    {
      v54 = v56;
      v55 = 0xC00000000LL;
      sub_11B1960((__int64)&v54, a3, -1, v30, v24, v25);
      v31 = (__int64)v54;
      v32 = (unsigned __int8 **)a2;
      do
      {
        v33 = **v32++ == 13;
        v31 += 4;
        *(_DWORD *)(v31 - 4) = -v33;
      }
      while ( v6 != v32 );
      v34 = *(__int64 **)(a1 + 112);
      sub_ACADE0(v43);
      v35 = sub_DFD330(v34);
      v36 = sub_DFBC30(*(__int64 **)(a1 + 112), 0, (__int64)v43, (__int64)v54, (unsigned int)v55, 0, 0, 0, v26, 1, 0);
      result = v36 + v35;
      if ( __OFADD__(v36, v35) )
      {
        result = 0x7FFFFFFFFFFFFFFFLL;
        if ( v36 <= 0 )
          result = 0x8000000000000000LL;
      }
      if ( v54 != v56 )
      {
        v49 = result;
        _libc_free((unsigned __int64)v54);
        result = v49;
      }
    }
    v15 = s2;
    goto LABEL_9;
  }
  v15 = s2;
  v37 = sub_2B0B400((unsigned __int8 **)s2, (__int64)s2 + 8 * v22);
  if ( v39 != v37 )
  {
    v19 = *(_QWORD *)(a1 + 184);
    v20 = *(_QWORD *)a1;
    if ( v38 != a3 )
    {
      result = sub_2B50660(v19, (__int64)v15, v38, 0, v20);
      goto LABEL_13;
    }
    v18 = 1;
    if ( v5 )
    {
      v45 = *(_QWORD *)(a1 + 184);
      v50 = *(_QWORD *)a1;
      v41 = memcmp((const void *)a2, v15, v5);
      v19 = v45;
      v20 = v50;
      v18 = v41 == 0;
    }
LABEL_12:
    result = sub_2B50660(v19, (__int64)v15, v46, v18, v20);
LABEL_13:
    v15 = s2;
    goto LABEL_9;
  }
  result = 0;
LABEL_9:
  if ( v15 != v53 )
  {
    v48 = result;
    _libc_free((unsigned __int64)v15);
    return v48;
  }
  return result;
}
