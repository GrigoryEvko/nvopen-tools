// Function: sub_11E8E10
// Address: 0x11e8e10
//
__int64 __fastcall sub_11E8E10(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  int v5; // edx
  int v6; // edx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 result; // rax
  int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  int v22; // r15d
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  char v27; // dl
  __int64 v28; // rdi
  _BYTE *v29; // r14
  _BYTE *v30; // rax
  __int64 *v31; // r14
  __int64 v32; // r12
  __int64 v33; // r15
  unsigned int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 **v38; // r15
  unsigned int v39; // r14d
  unsigned int v40; // eax
  unsigned __int64 v41; // rax
  size_t v42; // [rsp+0h] [rbp-90h]
  _QWORD **v43; // [rsp+8h] [rbp-88h]
  unsigned __int64 v44; // [rsp+8h] [rbp-88h]
  int v45; // [rsp+18h] [rbp-78h]
  void *s; // [rsp+20h] [rbp-70h] BYREF
  size_t n; // [rsp+28h] [rbp-68h]
  const char *v48; // [rsp+30h] [rbp-60h] BYREF
  char v49; // [rsp+50h] [rbp-40h]
  char v50; // [rsp+51h] [rbp-3Fh]

  sub_11E6850(a1, a2, a3, 0);
  v5 = *((_DWORD *)a2 + 1);
  s = 0;
  n = 0;
  if ( !(unsigned __int8)sub_98B0F0(*(_QWORD *)&a2[32 * (1LL - (v5 & 0x7FFFFFF))], &s, 1u) || *((_QWORD *)a2 + 2) )
    return 0;
  v6 = *a2;
  if ( v6 == 40 )
  {
    v7 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v7 = 0;
    if ( v6 != 85 )
    {
      v7 = 64;
      if ( v6 != 34 )
LABEL_54:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_11;
  v8 = sub_BD2BC0((__int64)a2);
  v10 = v8 + v9;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( !(unsigned int)(v10 >> 4) )
    {
LABEL_11:
      v11 = 0;
      goto LABEL_12;
    }
    goto LABEL_55;
  }
  if ( !(unsigned int)((v10 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_11;
  if ( (a2[7] & 0x80u) == 0 )
LABEL_55:
    BUG();
  v14 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v15 = sub_BD2BC0((__int64)a2);
  v11 = 32LL * (unsigned int)(*(_DWORD *)(v15 + v16 - 4) - v14);
LABEL_12:
  v12 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  if ( (unsigned int)((32 * v12 - 32 - v7 - v11) >> 5) == 2 )
  {
    if ( n )
    {
      v29 = s;
      v30 = memchr(s, 37, n);
      if ( v30 )
      {
        if ( v30 - v29 != -1 )
          return 0;
      }
    }
    v31 = *(__int64 **)(a1 + 24);
    v32 = *(_QWORD *)(a1 + 16);
    v33 = *(_QWORD *)&a2[-32 * v12];
    v43 = (_QWORD **)sub_B43CA0((__int64)a2);
    v42 = n;
    v34 = sub_97FA80(*v31, (__int64)v43);
    v35 = sub_BCCE00(*v43, v34);
    v36 = sub_ACD640(v35, v42, 0);
    result = sub_11CB6D0(*(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))], v36, v33, a3, v32, v31);
    if ( !result )
      return 0;
    goto LABEL_44;
  }
  if ( n != 2 || *(_BYTE *)s != 37 )
    return 0;
  v17 = *a2;
  if ( v17 == 40 )
  {
    v18 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v18 = 0;
    if ( v17 != 85 )
    {
      v18 = 64;
      if ( v17 != 34 )
        goto LABEL_54;
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v19 = sub_BD2BC0((__int64)a2);
    v21 = v19 + v20;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v21 >> 4) )
        goto LABEL_52;
    }
    else if ( (unsigned int)((v21 - sub_BD2BC0((__int64)a2)) >> 4) )
    {
      if ( (a2[7] & 0x80u) != 0 )
      {
        v22 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v23 = sub_BD2BC0((__int64)a2);
        v25 = 32LL * (unsigned int)(*(_DWORD *)(v23 + v24 - 4) - v22);
        goto LABEL_34;
      }
LABEL_52:
      BUG();
    }
  }
  v25 = 0;
LABEL_34:
  v26 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  if ( (unsigned int)((32 * v26 - 32 - v18 - v25) >> 5) <= 2 )
    return 0;
  v27 = *((_BYTE *)s + 1);
  if ( v27 == 99 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&a2[32 * (2 - v26)] + 8LL) + 8LL) != 12 )
      return 0;
    v37 = sub_BCD140(*(_QWORD **)(a3 + 72), *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL));
    v50 = 1;
    v38 = (__int64 **)v37;
    v49 = 3;
    v48 = "chari";
    v44 = *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    v39 = sub_BCB060(*(_QWORD *)(v44 + 8));
    v40 = sub_BCB060((__int64)v38);
    v41 = sub_11DB4B0((__int64 *)a3, v40 < v39 ? 38 : 40, v44, v38, (__int64)&v48, 0, v45, 0);
    result = sub_11CD1C0(v41, *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)], a3, *(__int64 **)(a1 + 24));
    if ( !result )
      return 0;
  }
  else
  {
    if ( v27 != 115 )
      return 0;
    v28 = *(_QWORD *)&a2[32 * (2 - v26)];
    if ( *(_BYTE *)(*(_QWORD *)(v28 + 8) + 8LL) != 14 )
      return 0;
    result = sub_11CB400(v28, *(_QWORD *)&a2[-32 * v26], a3, *(__int64 **)(a1 + 24));
    if ( !result )
      return 0;
  }
LABEL_44:
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *((_WORD *)a2 + 1) & 3;
  return result;
}
