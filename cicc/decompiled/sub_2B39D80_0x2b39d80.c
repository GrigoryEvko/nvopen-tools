// Function: sub_2B39D80
// Address: 0x2b39d80
//
__int64 __fastcall sub_2B39D80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, __int64 a6)
{
  unsigned __int8 **v6; // r10
  unsigned __int8 **v8; // r15
  __int64 v9; // r8
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int8 *v13; // rax
  unsigned __int8 v14; // si
  unsigned __int8 v15; // si
  unsigned __int8 v16; // si
  unsigned __int8 v17; // si
  __int64 v18; // r14
  unsigned int v19; // r13d
  char v21; // al
  unsigned __int8 **v22; // rsi
  unsigned int v23; // r12d
  __int64 v24; // r10
  unsigned int v25; // r8d
  unsigned __int8 *v26; // rdi
  unsigned __int8 v27; // al
  _DWORD *v28; // rdi
  __int64 v29; // r8
  __int64 v30; // rax
  char v31; // r9
  int v32; // edx
  unsigned int v33; // edx
  _DWORD *v34; // rcx
  unsigned int v35; // eax
  unsigned __int8 v36; // si
  unsigned __int8 v37; // si
  __int64 v38; // rdx
  unsigned int v39; // eax
  unsigned int v40; // eax
  unsigned int v41; // edx
  unsigned __int8 **v42; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v43; // [rsp+18h] [rbp-98h]
  unsigned int v44; // [rsp+20h] [rbp-90h]
  unsigned __int8 v45; // [rsp+28h] [rbp-88h]
  unsigned __int8 **v46; // [rsp+28h] [rbp-88h]
  __int64 v47; // [rsp+38h] [rbp-78h]
  void *s; // [rsp+40h] [rbp-70h] BYREF
  __int64 v49; // [rsp+48h] [rbp-68h]
  _DWORD v50[24]; // [rsp+50h] [rbp-60h] BYREF

  v6 = (unsigned __int8 **)a2;
  v8 = (unsigned __int8 **)(a2 + 8 * a3);
  v9 = (8 * a3) >> 5;
  v11 = (8 * a3) >> 3;
  if ( v9 <= 0 )
  {
    v12 = a2;
LABEL_69:
    if ( v11 != 2 )
    {
      if ( v11 != 3 )
      {
        if ( v11 != 1 )
          goto LABEL_72;
        goto LABEL_77;
      }
      v13 = *(unsigned __int8 **)v12;
      v36 = **(_BYTE **)v12;
      if ( v36 > 0x1Cu && (v36 == 90 || v36 == 93) )
        goto LABEL_16;
      v12 += 8;
    }
    v13 = *(unsigned __int8 **)v12;
    v37 = **(_BYTE **)v12;
    if ( v37 > 0x1Cu && (v37 == 90 || v37 == 93) )
      goto LABEL_16;
    v12 += 8;
LABEL_77:
    v13 = *(unsigned __int8 **)v12;
    v12 = **(unsigned __int8 **)v12;
    if ( (unsigned __int8)v12 > 0x1Cu && ((_BYTE)v12 == 90 || (_BYTE)v12 == 93) )
      goto LABEL_16;
LABEL_72:
    v13 = *v8;
    goto LABEL_16;
  }
  v12 = a2;
  v9 = a2 + 32 * v9;
  while ( 1 )
  {
    v13 = *(unsigned __int8 **)v12;
    v17 = **(_BYTE **)v12;
    if ( v17 > 0x1Cu && (v17 == 93 || v17 == 90) )
      break;
    v13 = *(unsigned __int8 **)(v12 + 8);
    v14 = *v13;
    if ( *v13 > 0x1Cu && (v14 == 90 || v14 == 93) )
      break;
    v13 = *(unsigned __int8 **)(v12 + 16);
    v15 = *v13;
    if ( *v13 > 0x1Cu && (v15 == 90 || v15 == 93) )
      break;
    v13 = *(unsigned __int8 **)(v12 + 24);
    v16 = *v13;
    if ( *v13 > 0x1Cu && (v16 == 90 || v16 == 93) )
      break;
    v12 += 32;
    if ( v9 == v12 )
    {
      v11 = ((__int64)v8 - v12) >> 3;
      goto LABEL_69;
    }
  }
LABEL_16:
  if ( (v13[7] & 0x40) != 0 )
  {
    v18 = **((_QWORD **)v13 - 1);
    *(_DWORD *)(a4 + 8) = 0;
    if ( *v13 != 93 )
    {
LABEL_18:
      v44 = a3;
      v19 = *(_DWORD *)(*(_QWORD *)(v18 + 8) + 32LL);
      goto LABEL_19;
    }
  }
  else
  {
    v12 = 32LL * (*((_DWORD *)v13 + 1) & 0x7FFFFFF);
    v18 = *(_QWORD *)&v13[-v12];
    *(_DWORD *)(a4 + 8) = 0;
    if ( *v13 != 93 )
      goto LABEL_18;
  }
  v44 = a3;
  v46 = v6;
  v19 = sub_2B2B880(a1, *(_QWORD *)(v18 + 8));
  if ( !v19 )
    return 0;
  if ( *(_BYTE *)v18 != 61 )
    return 0;
  if ( sub_B46500((unsigned __int8 *)v18) )
    return 0;
  if ( (*(_BYTE *)(v18 + 2) & 1) != 0 )
    return 0;
  v21 = sub_BD3610(v18, v44);
  LODWORD(a3) = v44;
  v6 = v46;
  if ( !v21 )
    return 0;
LABEL_19:
  v45 = (v19 != v44) & (a5 ^ 1);
  if ( v45 )
    return 0;
  v43 = (unsigned int)a3;
  s = v50;
  v49 = 0xC00000000LL;
  if ( (unsigned int)a3 > 0xC )
  {
    v42 = v6;
    LOBYTE(v9) = v19 != v44;
    sub_C8D5F0((__int64)&s, v50, (unsigned int)a3, 4u, v9, a6);
    memset(s, 255, 4 * v43);
    v6 = v42;
    LODWORD(v49) = v44;
  }
  else
  {
    if ( (_DWORD)a3 )
    {
      v35 = 4 * a3;
      if ( 4LL * (unsigned int)a3 )
      {
        if ( v35 >= 8 )
        {
          v38 = v35;
          v39 = v35 - 1;
          *(_QWORD *)((char *)&v50[-2] + v38) = -1;
          if ( v39 >= 8 )
          {
            v40 = v39 & 0xFFFFFFF8;
            v41 = 0;
            do
            {
              v12 = v41;
              v41 += 8;
              *(_QWORD *)((char *)v50 + v12) = -1;
            }
            while ( v41 < v40 );
          }
        }
        else if ( (v35 & 4) != 0 )
        {
          v50[0] = -1;
          v50[v35 / 4 - 1] = -1;
        }
        else if ( v35 )
        {
          LOBYTE(v50[0]) = -1;
        }
      }
    }
    LODWORD(v49) = v44;
  }
  if ( v6 == v8 )
  {
    v23 = v19;
    v29 = 1;
  }
  else
  {
    v22 = v6;
    v23 = v19;
    v24 = 0;
    v25 = 0;
    do
    {
      while ( 1 )
      {
        v26 = *v22;
        v27 = **v22;
        if ( v27 > 0x1Cu )
        {
          if ( (v26[7] & 0x40) != 0 )
          {
            v12 = *((_QWORD *)v26 - 1);
            if ( v18 != *(_QWORD *)v12 )
              goto LABEL_41;
          }
          else
          {
            v12 = (__int64)&v26[-32 * (*((_DWORD *)v26 + 1) & 0x7FFFFFF)];
            if ( v18 != *(_QWORD *)v12 )
              goto LABEL_41;
          }
          if ( v27 != 90 || (unsigned int)**((unsigned __int8 **)v26 - 4) - 12 > 1 )
          {
            v47 = sub_2B15730((__int64)v26);
            if ( !BYTE4(v47) )
              goto LABEL_41;
            if ( v19 > (unsigned int)v47 )
              break;
          }
        }
        ++v22;
        v24 += 4;
        if ( v8 == v22 )
          goto LABEL_51;
      }
      if ( v23 > (unsigned int)v47 )
        v23 = v47;
      if ( v25 < (unsigned int)v47 )
        v25 = v47;
      ++v22;
      *(_DWORD *)((char *)s + v24) = v47;
      v24 += 4;
    }
    while ( v8 != v22 );
LABEL_51:
    v29 = v25 + 1;
  }
  if ( (unsigned int)v29 - v23 > v44 )
  {
LABEL_41:
    v28 = s;
  }
  else
  {
    if ( (unsigned int)v29 <= v44 )
      v23 = 0;
    sub_2B39CB0(a4, v43, v44, v12, v29, a6);
    v28 = s;
    if ( v44 )
    {
      v30 = 0;
      v31 = 1;
      do
      {
        v32 = v28[v30];
        if ( v32 != -1 )
        {
          v33 = v32 - v23;
          v34 = (_DWORD *)(*(_QWORD *)a4 + 4LL * v33);
          if ( *v34 != v44 )
          {
            *(_DWORD *)(a4 + 8) = 0;
            goto LABEL_42;
          }
          *v34 = v30;
          v28 = s;
          v31 &= v33 == (_DWORD)v30;
        }
        ++v30;
      }
      while ( v30 != v44 );
      v45 = 0;
      if ( !v31 )
        goto LABEL_42;
    }
    *(_DWORD *)(a4 + 8) = 0;
    v45 = 1;
  }
LABEL_42:
  if ( v28 != v50 )
    _libc_free((unsigned __int64)v28);
  return v45;
}
