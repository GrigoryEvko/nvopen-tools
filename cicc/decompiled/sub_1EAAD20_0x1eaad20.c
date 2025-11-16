// Function: sub_1EAAD20
// Address: 0x1eaad20
//
__int64 __fastcall sub_1EAAD20(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // rdi
  __int64 *v4; // r14
  __int64 *i; // r12
  __int64 v6; // rdi
  __int64 v7; // rax
  char *v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  void (*v12)(void); // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rsi
  __int64 v17; // r12
  __int64 v18; // r13
  const void *v19; // r15
  __int64 v20; // rdx
  __int64 result; // rax
  int v22; // r13d
  unsigned int v23; // r12d
  unsigned int v24; // eax
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 (*v30)(); // rax
  _BYTE *v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // r15
  __int64 (*v35)(); // rcx
  int v36; // eax
  _BYTE *v37; // r13
  _BYTE *v38; // rax
  __int64 v39; // rdi
  __int64 (*v40)(); // rax
  _BYTE *v41; // rsi
  __int64 v42; // rdi
  __int64 *v43; // rdi
  __int64 v44; // rax
  void (*v45)(); // rdx
  __int64 (*v46)(); // rax
  __int64 v47; // r15
  _BYTE *v48; // r15
  void (*v49)(void); // rax
  int v50; // r13d
  int j; // r15d
  __int64 v52; // r12
  __int64 v53; // rax
  char *v54; // r13
  signed __int64 v55; // rdx
  char *v56; // rax
  __int64 v57; // rsi
  char v58; // [rsp+Bh] [rbp-75h]
  unsigned int v59; // [rsp+Ch] [rbp-74h]
  __int64 v60; // [rsp+10h] [rbp-70h]
  unsigned int v61; // [rsp+18h] [rbp-68h]
  char v62; // [rsp+18h] [rbp-68h]
  __int64 v63; // [rsp+18h] [rbp-68h]
  __int64 v64; // [rsp+28h] [rbp-58h] BYREF
  _BYTE *v65; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v66; // [rsp+38h] [rbp-48h]
  _BYTE *v67; // [rsp+40h] [rbp-40h]

  v2 = a1 + 48;
  sub_1F0A020(a1, *(_QWORD *)(a1 + 2224), 0, 0, 0, 0);
  v3 = *(_QWORD *)(a1 + 2216);
  if ( v3
    && (*(unsigned int (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD, __int64))(*(_QWORD *)v3 + 24LL))(
         v3,
         v2,
         *(_QWORD *)(a1 + 928),
         *(_QWORD *)(a1 + 936),
         *(unsigned int *)(a1 + 2280),
         a1 + 2000) )
  {
    sub_1F013F0(a1);
    sub_1F0A020(a1, *(_QWORD *)(a1 + 2224), 0, 0, 0, 0);
  }
  v4 = *(__int64 **)(a1 + 2264);
  for ( i = *(__int64 **)(a1 + 2256); v4 != i; ++i )
  {
    v6 = *i;
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 24LL))(v6, a1);
  }
  v7 = *(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 2120) = v2;
  v8 = *(char **)(a1 + 2136);
  LODWORD(v65) = 0;
  v9 = 0xF0F0F0F0F0F0F0F1LL * (v7 >> 4);
  v10 = *(_QWORD *)(a1 + 2128);
  v11 = (__int64)&v8[-v10] >> 2;
  if ( v9 > v11 )
  {
    sub_1CFD340(a1 + 2128, v8, v9 - v11, &v65);
  }
  else if ( v9 < v11 )
  {
    v56 = (char *)(v10 + 4 * v9);
    if ( v8 != v56 )
      *(_QWORD *)(a1 + 2136) = v56;
  }
  v12 = *(void (**)(void))(**(_QWORD **)(a1 + 2208) + 32LL);
  if ( v12 != nullsub_678 )
    v12();
  sub_1EAABB0(a1, a1 + 72);
  v13 = *(_QWORD *)(a1 + 48);
  v14 = *(_QWORD *)(a1 + 56) - v13;
  v15 = 0xF0F0F0F0F0F0F0F1LL * (v14 >> 4);
  v16 = v15;
  if ( (_DWORD)v15 )
  {
    v17 = 0;
    v18 = 272LL * (unsigned int)v15;
    do
    {
      while ( *(_DWORD *)(v13 + v17 + 208) || (*(_BYTE *)(v13 + v17 + 229) & 2) != 0 )
      {
        v17 += 272;
        if ( v18 == v17 )
          goto LABEL_16;
      }
      sub_20F9700(a1 + 2104);
      *(_BYTE *)(*(_QWORD *)(a1 + 48) + v17 + 229) |= 2u;
      v17 += 272;
      v13 = *(_QWORD *)(a1 + 48);
    }
    while ( v18 != v17 );
LABEL_16:
    v14 = *(_QWORD *)(a1 + 56) - v13;
    v16 = 0xF0F0F0F0F0F0F0F1LL * (v14 >> 4);
  }
  v65 = 0;
  v66 = 0;
  v67 = 0;
  if ( v14 < 0 )
    sub_4262D8((__int64)"vector::reserve");
  v19 = *(const void **)(a1 + 2232);
  if ( v16 <= (__int64)(*(_QWORD *)(a1 + 2248) - (_QWORD)v19) >> 3 )
    goto LABEL_19;
  v63 = 8 * v16;
  v52 = *(_QWORD *)(a1 + 2240) - (_QWORD)v19;
  if ( v16 )
  {
    v53 = sub_22077B0(8 * v16);
    v19 = *(const void **)(a1 + 2232);
    v54 = (char *)v53;
    v55 = *(_QWORD *)(a1 + 2240) - (_QWORD)v19;
    if ( v55 <= 0 )
      goto LABEL_97;
LABEL_103:
    memmove(v54, v19, v55);
    v57 = *(_QWORD *)(a1 + 2248) - (_QWORD)v19;
LABEL_104:
    j_j___libc_free_0(v19, v57);
    goto LABEL_98;
  }
  v55 = *(_QWORD *)(a1 + 2240) - (_QWORD)v19;
  v54 = 0;
  if ( v52 > 0 )
    goto LABEL_103;
LABEL_97:
  if ( v19 )
  {
    v57 = *(_QWORD *)(a1 + 2248) - (_QWORD)v19;
    goto LABEL_104;
  }
LABEL_98:
  *(_QWORD *)(a1 + 2232) = v54;
  *(_QWORD *)(a1 + 2240) = &v54[v52];
  *(_QWORD *)(a1 + 2248) = &v54[v63];
LABEL_19:
  v58 = 0;
  v59 = 0;
  while ( 1 )
  {
    v20 = *(_QWORD *)(a1 + 2192);
    result = *(_QWORD *)(a1 + 2184);
    if ( *(_QWORD *)(a1 + 2160) == *(_QWORD *)(a1 + 2152) )
      break;
    v22 = (v20 - result) >> 3;
    if ( !v22 )
      goto LABEL_33;
LABEL_22:
    v61 = -1;
    v23 = 0;
    while ( 1 )
    {
      v25 = v23;
      v26 = *(_QWORD *)(result + 8LL * v23);
      v27 = v26;
      if ( (*(_BYTE *)(v26 + 236) & 1) == 0 )
        break;
      if ( *(_DWORD *)(v26 + 240) <= v59 )
        goto LABEL_31;
LABEL_24:
      ++v23;
      if ( (*(_BYTE *)(v27 + 236) & 1) != 0 )
      {
        v24 = *(_DWORD *)(v27 + 240);
        if ( v61 <= v24 )
          goto LABEL_27;
        goto LABEL_26;
      }
      sub_1F01DD0(v27);
      if ( *(_DWORD *)(v27 + 240) < v61 )
      {
        v47 = *(_QWORD *)(*(_QWORD *)(a1 + 2184) + 8 * v25);
        if ( (*(_BYTE *)(v47 + 236) & 1) == 0 )
          sub_1F01DD0(v47);
        v24 = *(_DWORD *)(v47 + 240);
LABEL_26:
        v61 = v24;
      }
LABEL_27:
      if ( v22 == v23 )
        goto LABEL_32;
LABEL_28:
      result = *(_QWORD *)(a1 + 2184);
    }
    v60 = v26;
    sub_1F01DD0(v26);
    v27 = *(_QWORD *)(*(_QWORD *)(a1 + 2184) + 8LL * v23);
    if ( *(_DWORD *)(v60 + 240) > v59 )
      goto LABEL_24;
LABEL_31:
    --v22;
    sub_20F9700(a1 + 2104);
    v28 = *(_QWORD *)(*(_QWORD *)(a1 + 2184) + 8LL * v23);
    *(_BYTE *)(v28 + 229) |= 2u;
    *(_QWORD *)(*(_QWORD *)(a1 + 2184) + 8LL * v23) = *(_QWORD *)(*(_QWORD *)(a1 + 2192) - 8LL);
    *(_QWORD *)(a1 + 2192) -= 8LL;
    if ( v22 != v23 )
      goto LABEL_28;
LABEL_32:
    if ( *(_QWORD *)(a1 + 2160) != *(_QWORD *)(a1 + 2152) )
    {
LABEL_33:
      v62 = 0;
      v29 = 0;
      while ( 1 )
      {
        v32 = sub_20F9440(a1 + 2104);
        v33 = *(_QWORD *)(a1 + 2208);
        v64 = v32;
        v34 = v32;
        v35 = *(__int64 (**)())(*(_QWORD *)v33 + 24LL);
        if ( v35 == sub_1D00B90 )
        {
          v30 = *(__int64 (**)())(*(_QWORD *)v33 + 72LL);
          if ( v30 == sub_1EA9B50 )
            goto LABEL_63;
        }
        else
        {
          v36 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v35)(v33, v32, 0);
          if ( v36 )
          {
            v31 = v66;
            v62 |= v36 == 2;
            if ( v66 == v67 )
              goto LABEL_45;
LABEL_38:
            if ( v31 )
            {
              *(_QWORD *)v31 = v64;
              v31 = v66;
            }
            v66 = v31 + 8;
            goto LABEL_41;
          }
          v33 = *(_QWORD *)(a1 + 2208);
          v34 = v64;
          v30 = *(__int64 (**)())(*(_QWORD *)v33 + 72LL);
          if ( v30 == sub_1EA9B50 )
          {
LABEL_63:
            if ( v29 )
            {
LABEL_64:
              if ( v34 )
              {
                v29 = v34;
                sub_20F9700(a1 + 2104);
              }
LABEL_47:
              v37 = v66;
              v38 = v65;
              if ( v66 != v65 )
                goto LABEL_71;
              goto LABEL_48;
            }
            goto LABEL_90;
          }
        }
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v30)(v33, v34) )
        {
          v34 = v64;
          if ( v29 )
            goto LABEL_64;
LABEL_90:
          v37 = v66;
          v38 = v65;
          v29 = v34;
          if ( v66 != v65 )
            goto LABEL_71;
          goto LABEL_75;
        }
        if ( v29 )
        {
          v31 = v66;
          if ( v66 != v67 )
            goto LABEL_38;
LABEL_45:
          sub_1CFD630((__int64)&v65, v31, &v64);
          if ( *(_QWORD *)(a1 + 2160) == *(_QWORD *)(a1 + 2152) )
          {
LABEL_46:
            if ( v29 )
              goto LABEL_47;
            v37 = v66;
            v38 = v65;
            if ( v65 != v66 )
              goto LABEL_71;
            goto LABEL_76;
          }
        }
        else
        {
          v29 = v64;
LABEL_41:
          if ( *(_QWORD *)(a1 + 2160) == *(_QWORD *)(a1 + 2152) )
            goto LABEL_46;
        }
      }
    }
LABEL_69:
    v37 = v66;
    v38 = v65;
    if ( v65 == v66 )
    {
LABEL_78:
      v49 = *(void (**)(void))(**(_QWORD **)(a1 + 2208) + 80LL);
      if ( v49 != nullsub_683 )
        v49();
      goto LABEL_80;
    }
    v62 = 0;
    v29 = 0;
LABEL_71:
    v48 = v38;
    do
    {
      v48 += 8;
      sub_20F9700(a1 + 2104);
    }
    while ( v37 != v48 );
    if ( v65 != v66 )
      v66 = v65;
LABEL_75:
    if ( v29 )
    {
LABEL_48:
      v39 = *(_QWORD *)(a1 + 2208);
      v40 = *(__int64 (**)())(*(_QWORD *)v39 + 56LL);
      if ( v40 != sub_1EA9B40 )
      {
        v50 = ((__int64 (__fastcall *)(__int64, __int64))v40)(v39, v29);
        if ( v50 )
        {
          for ( j = 0; j != v50; ++j )
            sub_1EAAC80((_QWORD *)a1);
        }
      }
      v64 = v29;
      v41 = *(_BYTE **)(a1 + 2240);
      if ( v41 == *(_BYTE **)(a1 + 2248) )
      {
        sub_1CFD630(a1 + 2232, v41, &v64);
        v42 = v64;
      }
      else
      {
        if ( v41 )
        {
          *(_QWORD *)v41 = v29;
          v41 = *(_BYTE **)(a1 + 2240);
        }
        v42 = v29;
        *(_QWORD *)(a1 + 2240) = v41 + 8;
      }
      sub_1F01F20(v42, v59);
      sub_1EAABB0(a1, v64);
      *(_BYTE *)(v64 + 229) |= 4u;
      sub_20F95D0(a1 + 2104);
      v43 = *(__int64 **)(a1 + 2208);
      v44 = *v43;
      v45 = *(void (**)())(*v43 + 40);
      if ( v45 != nullsub_679 )
      {
        ((void (__fastcall *)(__int64 *, __int64))v45)(v43, v29);
        v44 = **(_QWORD **)(a1 + 2208);
      }
      v46 = *(__int64 (**)())(v44 + 16);
      v58 = 1;
      if ( v46 != sub_1D00B80 && (unsigned __int8)v46() )
        goto LABEL_78;
    }
    else
    {
LABEL_76:
      if ( v58 || !v62 )
        goto LABEL_78;
      sub_1EAAC80((_QWORD *)a1);
LABEL_80:
      ++v59;
      v58 = 0;
    }
  }
  if ( v20 != result )
  {
    v22 = (v20 - result) >> 3;
    if ( !v22 )
      goto LABEL_69;
    goto LABEL_22;
  }
  if ( v65 )
    result = j_j___libc_free_0(v65, v67 - v65);
  *(_QWORD *)(a1 + 2120) = 0;
  return result;
}
