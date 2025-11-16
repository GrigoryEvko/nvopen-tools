// Function: sub_D1C730
// Address: 0xd1c730
//
__int64 __fastcall sub_D1C730(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, __int64 a4)
{
  char v7; // al
  int v8; // edx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int8 **v17; // r12
  __int64 v18; // rdx
  unsigned __int8 **v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int8 **v22; // r14
  unsigned __int8 **v23; // rdi
  __int64 v24; // rdx
  unsigned __int8 **v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned __int8 **v28; // rax
  unsigned __int8 **v29; // rdx
  unsigned __int8 **v30; // r15
  __int64 v31; // r12
  __int64 v32; // rax
  unsigned __int8 **v33; // rbx
  unsigned __int8 *v34; // rax
  unsigned __int8 *v35; // rax
  unsigned __int8 *v36; // rax
  unsigned __int8 *v37; // rax
  unsigned __int8 *v38; // rax
  unsigned __int8 **v39; // r15
  unsigned __int8 **v40; // r15
  unsigned __int8 **v41; // r15
  unsigned __int8 *v42; // rax
  unsigned __int8 *v43; // rax
  unsigned __int8 v44; // [rsp+7h] [rbp-109h]
  unsigned __int8 **v46; // [rsp+10h] [rbp-100h]
  __int64 *v47; // [rsp+18h] [rbp-F8h]
  __int64 v48; // [rsp+30h] [rbp-E0h]
  __int64 *v49; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v51; // [rsp+48h] [rbp-C8h]
  unsigned __int8 **v52; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+58h] [rbp-B8h]
  _BYTE v54[32]; // [rsp+60h] [rbp-B0h] BYREF
  unsigned __int8 *v55; // [rsp+80h] [rbp-90h] BYREF
  __int64 v56; // [rsp+88h] [rbp-88h]
  __int64 v57; // [rsp+90h] [rbp-80h]
  __int64 v58; // [rsp+98h] [rbp-78h]
  __int64 v59; // [rsp+A0h] [rbp-70h]
  __int64 v60; // [rsp+A8h] [rbp-68h]
  unsigned __int8 *v61; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v62; // [rsp+B8h] [rbp-58h]
  __int64 v63; // [rsp+C0h] [rbp-50h]
  __int64 v64; // [rsp+C8h] [rbp-48h]
  __int64 v65; // [rsp+D0h] [rbp-40h]
  __int64 v66; // [rsp+D8h] [rbp-38h]

  v44 = 0;
  if ( sub_B49E00((__int64)a2) )
    return v44;
  v7 = sub_B49E20((__int64)a2);
  v8 = *a2;
  v44 = v7 == 0 ? 3 : 1;
  if ( v8 == 40 )
  {
    v9 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v9 = -32;
    if ( v8 != 85 )
    {
      v9 = -96;
      if ( v8 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_14;
  v10 = sub_BD2BC0((__int64)a2);
  v12 = v10 + v11;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v12 >> 4) )
      goto LABEL_90;
  }
  else if ( (unsigned int)((v12 - sub_BD2BC0((__int64)a2)) >> 4) )
  {
    if ( (a2[7] & 0x80u) != 0 )
    {
      v13 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v14 = sub_BD2BC0((__int64)a2);
      v9 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
      goto LABEL_14;
    }
LABEL_90:
    BUG();
  }
LABEL_14:
  v47 = (__int64 *)&a2[v9];
  v16 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  if ( &a2[-v16] == &a2[v9] )
    return 0;
  v49 = (__int64 *)&a2[-v16];
  while ( 1 )
  {
    v52 = (unsigned __int8 **)v54;
    v53 = 0x400000000LL;
    sub_98B4D0(*v49, (__int64)&v52, 0, 6u);
    v17 = v52;
    v18 = 8LL * (unsigned int)v53;
    v19 = &v52[(unsigned __int64)v18 / 8];
    v20 = v18 >> 3;
    v21 = v18 >> 5;
    if ( v21 )
    {
      v22 = &v52[4 * v21];
      while ( (unsigned __int8)sub_CF7060(*v17) )
      {
        if ( !(unsigned __int8)sub_CF7060(v17[1]) )
        {
          ++v17;
          goto LABEL_23;
        }
        if ( !(unsigned __int8)sub_CF7060(v17[2]) )
        {
          v17 += 2;
          goto LABEL_23;
        }
        if ( !(unsigned __int8)sub_CF7060(v17[3]) )
        {
          v17 += 3;
          goto LABEL_23;
        }
        v17 += 4;
        if ( v22 == v17 )
        {
          v20 = v19 - v17;
          goto LABEL_49;
        }
      }
      goto LABEL_23;
    }
LABEL_49:
    if ( v20 == 2 )
      goto LABEL_72;
    if ( v20 == 3 )
    {
      if ( !(unsigned __int8)sub_CF7060(*v17) )
        goto LABEL_23;
      ++v17;
LABEL_72:
      if ( !(unsigned __int8)sub_CF7060(*v17) )
        goto LABEL_23;
      ++v17;
      goto LABEL_52;
    }
    if ( v20 != 1 )
      goto LABEL_24;
LABEL_52:
    if ( (unsigned __int8)sub_CF7060(*v17) )
      goto LABEL_24;
LABEL_23:
    if ( v19 == v17 )
      goto LABEL_24;
    v30 = v52;
    v31 = 8LL * (unsigned int)v53;
    v46 = &v52[(unsigned __int64)v31 / 8];
    v32 = v31 >> 3;
    v48 = v31 >> 5;
    if ( v31 >> 5 )
      break;
LABEL_75:
    if ( v32 != 2 )
    {
      if ( v32 != 3 )
      {
        if ( v32 != 1 )
          goto LABEL_24;
        goto LABEL_78;
      }
      v42 = *v30;
      v25 = &v55;
      v61 = a3;
      v62 = -1;
      v63 = 0;
      v64 = 0;
      v65 = 0;
      v66 = 0;
      v55 = v42;
      v56 = -1;
      v57 = 0;
      v58 = 0;
      v59 = 0;
      v60 = 0;
      if ( sub_D1C200(a1, &v55, &v61, a4, (__int64)a2) )
        goto LABEL_44;
      ++v30;
    }
    v43 = *v30;
    v25 = &v55;
    v61 = a3;
    v62 = -1;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v55 = v43;
    v56 = -1;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    if ( sub_D1C200(a1, &v55, &v61, a4, (__int64)a2) )
      goto LABEL_44;
    ++v30;
LABEL_78:
    v38 = *v30;
    v25 = &v55;
    v61 = a3;
    v62 = -1;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v55 = v38;
    v56 = -1;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    if ( sub_D1C200(a1, &v55, &v61, a4, (__int64)a2) )
      goto LABEL_44;
LABEL_24:
    v23 = v52;
    v24 = 8LL * (unsigned int)v53;
    v25 = &v52[(unsigned __int64)v24 / 8];
    v26 = v24 >> 3;
    v27 = v24 >> 5;
    if ( v27 )
    {
      v28 = v52;
      v29 = &v52[4 * v27];
      while ( a3 != *v28 )
      {
        if ( a3 == v28[1] )
        {
          ++v28;
          goto LABEL_31;
        }
        if ( a3 == v28[2] )
        {
          v28 += 2;
          goto LABEL_31;
        }
        if ( a3 == v28[3] )
        {
          v28 += 3;
          goto LABEL_31;
        }
        v28 += 4;
        if ( v29 == v28 )
        {
          v26 = v25 - v28;
          goto LABEL_61;
        }
      }
      goto LABEL_31;
    }
    v28 = v52;
LABEL_61:
    if ( v26 == 2 )
      goto LABEL_68;
    if ( v26 != 3 )
    {
      if ( v26 != 1 )
        goto LABEL_32;
      goto LABEL_64;
    }
    if ( a3 != *v28 )
    {
      ++v28;
LABEL_68:
      if ( a3 != *v28 )
      {
        ++v28;
LABEL_64:
        if ( a3 != *v28 )
          goto LABEL_32;
      }
    }
LABEL_31:
    if ( v25 != v28 )
      goto LABEL_46;
LABEL_32:
    if ( v52 != (unsigned __int8 **)v54 )
      _libc_free(v52, v25);
    v49 += 4;
    if ( v47 == v49 )
      return 0;
  }
  v51 = a3;
  v33 = v52;
  while ( 1 )
  {
    v37 = *v33;
    v25 = &v55;
    v62 = -1;
    v61 = v51;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v55 = v37;
    v56 = -1;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    if ( sub_D1C200(a1, &v55, &v61, a4, (__int64)a2) )
    {
      v30 = v33;
      a3 = v51;
      goto LABEL_44;
    }
    v25 = &v55;
    v34 = v33[1];
    v62 = -1;
    v61 = v51;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v55 = v34;
    v56 = -1;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    if ( sub_D1C200(a1, &v55, &v61, a4, (__int64)a2) )
    {
      v39 = v33;
      a3 = v51;
      v30 = v39 + 1;
      goto LABEL_44;
    }
    v35 = v33[2];
    v62 = -1;
    v61 = v51;
    v25 = &v55;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v55 = v35;
    v56 = -1;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    if ( sub_D1C200(a1, &v55, &v61, a4, (__int64)a2) )
    {
      v40 = v33;
      a3 = v51;
      v30 = v40 + 2;
      goto LABEL_44;
    }
    v36 = v33[3];
    v62 = -1;
    v61 = v51;
    v25 = &v55;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v55 = v36;
    v56 = -1;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    if ( sub_D1C200(a1, &v55, &v61, a4, (__int64)a2) )
      break;
    v33 += 4;
    if ( !--v48 )
    {
      v30 = v33;
      a3 = v51;
      v32 = v46 - v30;
      goto LABEL_75;
    }
  }
  v41 = v33;
  a3 = v51;
  v30 = v41 + 3;
LABEL_44:
  if ( v30 == v46 )
    goto LABEL_24;
  v23 = v52;
LABEL_46:
  if ( v23 != (unsigned __int8 **)v54 )
    _libc_free(v23, v25);
  return v44;
}
