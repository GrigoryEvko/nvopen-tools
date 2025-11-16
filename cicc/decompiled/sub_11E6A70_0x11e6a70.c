// Function: sub_11E6A70
// Address: 0x11e6a70
//
__int64 __fastcall sub_11E6A70(__int64 a1, unsigned __int8 *a2, __int64 *a3)
{
  int v5; // eax
  size_t v6; // r14
  __int64 v7; // r10
  __int64 **v9; // r11
  _BYTE *v10; // r15
  size_t v11; // rcx
  int v12; // edx
  __int64 v13; // r15
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  int v19; // r14d
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdi
  __int64 v24; // rax
  int v25; // edx
  __int64 v26; // r15
  unsigned int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rax
  int v32; // r14d
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rdi
  char v38; // al
  signed __int64 v39; // rdx
  __int64 v40; // rax
  _BYTE *v41; // rax
  int v42; // edx
  __int64 v43; // r15
  unsigned int v44; // eax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r14
  __int64 v48; // rax
  int v49; // r14d
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  unsigned __int64 v53; // r15
  __int64 v54; // rax
  _BYTE *v55; // rax
  unsigned int v56; // r14d
  unsigned int v57; // eax
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // [rsp+8h] [rbp-98h]
  __int64 v61; // [rsp+10h] [rbp-90h]
  __int64 v62; // [rsp+10h] [rbp-90h]
  __int64 v63; // [rsp+10h] [rbp-90h]
  __int64 v64; // [rsp+10h] [rbp-90h]
  __int64 v65; // [rsp+10h] [rbp-90h]
  __int64 **v66; // [rsp+10h] [rbp-90h]
  __int64 v67; // [rsp+18h] [rbp-88h]
  __int64 v68; // [rsp+18h] [rbp-88h]
  __int64 **v69; // [rsp+18h] [rbp-88h]
  __int64 **v70; // [rsp+18h] [rbp-88h]
  __int64 v71; // [rsp+18h] [rbp-88h]
  __int64 **v72; // [rsp+18h] [rbp-88h]
  __int64 **v73; // [rsp+18h] [rbp-88h]
  size_t v74; // [rsp+18h] [rbp-88h]
  __int64 **v75; // [rsp+18h] [rbp-88h]
  void *s; // [rsp+20h] [rbp-80h] BYREF
  size_t n; // [rsp+28h] [rbp-78h]
  char *v78; // [rsp+30h] [rbp-70h] BYREF
  __int64 v79; // [rsp+38h] [rbp-68h]
  _QWORD v80[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v81; // [rsp+60h] [rbp-40h]

  v5 = *((_DWORD *)a2 + 1);
  s = 0;
  n = 0;
  if ( !(unsigned __int8)sub_98B0F0(*(_QWORD *)&a2[-32 * (v5 & 0x7FFFFFF)], &s, 1u) )
    return 0;
  v6 = n;
  v7 = *((_QWORD *)a2 + 2);
  if ( !n )
  {
    if ( v7 )
      return sub_AD64C0(*((_QWORD *)a2 + 1), 0, 0);
    return (__int64)a2;
  }
  if ( v7 )
    return 0;
  v9 = (__int64 **)*((_QWORD *)a2 + 1);
  v10 = s;
  if ( n == 1 )
    goto LABEL_30;
  if ( n != 2 )
    goto LABEL_9;
  if ( *(_WORD *)s == 9509 )
  {
LABEL_30:
    v24 = sub_AD64C0(*((_QWORD *)a2 + 1), *(unsigned __int8 *)s, 0);
    v7 = sub_11CCAE0(v24, (__int64)a3, *(__int64 **)(a1 + 24));
    if ( !v7 )
      return 0;
    goto LABEL_31;
  }
  if ( *(_WORD *)s != 29477 )
  {
LABEL_9:
    v11 = n - 1;
    if ( *((char *)s + n - 1) != 10 )
      goto LABEL_10;
    goto LABEL_79;
  }
  v25 = *a2;
  if ( v25 == 40 )
  {
    v61 = *((_QWORD *)a2 + 2);
    v69 = (__int64 **)*((_QWORD *)a2 + 1);
    v27 = sub_B491D0((__int64)a2);
    v9 = v69;
    v7 = v61;
    v26 = 32LL * v27;
  }
  else
  {
    v26 = 0;
    if ( v25 != 85 )
    {
      v26 = 64;
      if ( v25 != 34 )
LABEL_94:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v62 = v7;
    v70 = v9;
    v28 = sub_BD2BC0((__int64)a2);
    v9 = v70;
    v7 = v62;
    v30 = v28 + v29;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v30 >> 4) )
        goto LABEL_90;
    }
    else
    {
      v31 = sub_BD2BC0((__int64)a2);
      v9 = v70;
      v7 = v62;
      if ( (unsigned int)((v30 - v31) >> 4) )
      {
        if ( (a2[7] & 0x80u) != 0 )
        {
          v32 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
          if ( (a2[7] & 0x80u) == 0 )
            BUG();
          v33 = sub_BD2BC0((__int64)a2);
          v7 = v62;
          v9 = v70;
          v35 = 32LL * (unsigned int)(*(_DWORD *)(v33 + v34 - 4) - v32);
          goto LABEL_46;
        }
LABEL_90:
        BUG();
      }
    }
  }
  v35 = 0;
LABEL_46:
  v36 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  if ( (unsigned int)((32 * v36 - 32 - v26 - v35) >> 5) > 1 )
  {
    v71 = v7;
    v63 = (__int64)v9;
    v78 = 0;
    v37 = *(_QWORD *)&a2[32 * (1 - v36)];
    v79 = 0;
    v38 = sub_98B0F0(v37, &v78, 1u);
    v7 = v71;
    if ( !v38 )
      return v7;
    if ( !v79 )
      return (__int64)a2;
    if ( v79 == 1 )
    {
      v59 = sub_AD64C0(v63, (unsigned __int8)*v78, 0);
      v41 = (_BYTE *)sub_11CCAE0(v59, (__int64)a3, *(__int64 **)(a1 + 24));
      v7 = v71;
      if ( !v41 )
        return v7;
    }
    else
    {
      v39 = v79 - 1;
      if ( v78[v79 - 1] != 10 )
        return v7;
      --v79;
      v80[0] = "str";
      v81 = 259;
      v40 = sub_B33830((__int64)a3, v78, v39, (__int64)v80, 0, 0, 1);
      v41 = (_BYTE *)sub_11CB1C0(v40, (__int64)a3, *(__int64 **)(a1 + 24));
      v7 = v71;
      if ( !v41 )
        return v7;
    }
    v7 = (__int64)v41;
    if ( *v41 != 85 )
      return v7;
    goto LABEL_32;
  }
  v6 = n;
  v10 = s;
  v11 = n - 1;
  if ( *((char *)s + n - 1) != 10 )
    goto LABEL_10;
  if ( !n )
    goto LABEL_77;
LABEL_79:
  v74 = v11;
  v60 = v7;
  v66 = v9;
  v55 = memchr(v10, 37, v6);
  if ( !v55 || (v9 = v66, v7 = v60, v55 - v10 == -1) )
  {
    v6 = v74;
LABEL_77:
    n = v6;
    v80[0] = "str";
    v81 = 259;
    v54 = sub_B33830((__int64)a3, (char *)s, v6, (__int64)v80, 0, 0, 1);
    v7 = sub_11CB1C0(v54, (__int64)a3, *(__int64 **)(a1 + 24));
    if ( !v7 )
      return 0;
    goto LABEL_31;
  }
LABEL_10:
  if ( v6 == 2 )
  {
    if ( *(_WORD *)v10 != 25381 )
      return v7;
    v42 = *a2;
    if ( v42 == 40 )
    {
      v64 = v7;
      v72 = v9;
      v44 = sub_B491D0((__int64)a2);
      v7 = v64;
      v9 = v72;
      v43 = 32LL * v44;
    }
    else
    {
      v43 = 0;
      if ( v42 != 85 )
      {
        v43 = 64;
        if ( v42 != 34 )
          goto LABEL_94;
      }
    }
    if ( (a2[7] & 0x80u) != 0 )
    {
      v65 = v7;
      v73 = v9;
      v45 = sub_BD2BC0((__int64)a2);
      v9 = v73;
      v7 = v65;
      v47 = v45 + v46;
      if ( (a2[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v47 >> 4) )
          goto LABEL_95;
      }
      else
      {
        v48 = sub_BD2BC0((__int64)a2);
        v9 = v73;
        v7 = v65;
        if ( (unsigned int)((v47 - v48) >> 4) )
        {
          if ( (a2[7] & 0x80u) != 0 )
          {
            v49 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
            if ( (a2[7] & 0x80u) == 0 )
              BUG();
            v50 = sub_BD2BC0((__int64)a2);
            v7 = v65;
            v9 = v73;
            v52 = 32LL * (unsigned int)(*(_DWORD *)(v50 + v51 - 4) - v49);
            goto LABEL_68;
          }
LABEL_95:
          BUG();
        }
      }
    }
    v52 = 0;
LABEL_68:
    if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v43 - v52) >> 5) <= 1
      || (v53 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
          *(_BYTE *)(*(_QWORD *)(v53 + 8) + 8LL) != 12) )
    {
      v10 = s;
      v6 = n;
      goto LABEL_11;
    }
    v75 = v9;
    v81 = 257;
    v56 = sub_BCB060(*(_QWORD *)(v53 + 8));
    v57 = sub_BCB060((__int64)v75);
    v58 = sub_11DB4B0(a3, (unsigned int)(v56 <= v57) + 38, v53, v75, (__int64)v80, 0, (int)v78, 0);
    v7 = sub_11CCAE0(v58, (__int64)a3, *(__int64 **)(a1 + 24));
    if ( !v7 )
      return 0;
LABEL_31:
    if ( *(_BYTE *)v7 != 85 )
      return v7;
LABEL_32:
    *(_WORD *)(v7 + 2) = *(_WORD *)(v7 + 2) & 0xFFFC | *((_WORD *)a2 + 1) & 3;
    return v7;
  }
LABEL_11:
  if ( v6 != 3 || *(_WORD *)v10 != 29477 || v10[2] != 10 )
    return v7;
  v12 = *a2;
  if ( v12 == 40 )
  {
    v67 = v7;
    v14 = sub_B491D0((__int64)a2);
    v7 = v67;
    v13 = 32LL * v14;
  }
  else
  {
    v13 = 0;
    if ( v12 != 85 )
    {
      v13 = 64;
      if ( v12 != 34 )
        goto LABEL_94;
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_55;
  v68 = v7;
  v15 = sub_BD2BC0((__int64)a2);
  v7 = v68;
  v17 = v15 + v16;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v17 >> 4) )
LABEL_93:
      BUG();
LABEL_55:
    v22 = 0;
    goto LABEL_25;
  }
  v18 = sub_BD2BC0((__int64)a2);
  v7 = v68;
  if ( !(unsigned int)((v17 - v18) >> 4) )
    goto LABEL_55;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_93;
  v19 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v20 = sub_BD2BC0((__int64)a2);
  v7 = v68;
  v22 = 32LL * (unsigned int)(*(_DWORD *)(v20 + v21 - 4) - v19);
LABEL_25:
  if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v13 - v22) >> 5) > 1 )
  {
    v23 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    if ( *(_BYTE *)(*(_QWORD *)(v23 + 8) + 8LL) == 14 )
    {
      v7 = sub_11CB1C0(v23, (__int64)a3, *(__int64 **)(a1 + 24));
      if ( !v7 )
        return 0;
      goto LABEL_31;
    }
  }
  return v7;
}
