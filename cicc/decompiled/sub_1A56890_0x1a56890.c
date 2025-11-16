// Function: sub_1A56890
// Address: 0x1a56890
//
void __fastcall sub_1A56890(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r14
  _BYTE *v7; // r9
  int v8; // r10d
  __int64 v9; // r12
  _QWORD *v10; // rcx
  __int64 *v11; // rdx
  int v12; // r13d
  unsigned int v13; // esi
  __int64 *v14; // rax
  __int64 v15; // r15
  __int64 *v16; // rsi
  __int64 *v17; // rax
  __int64 *v18; // r15
  char v19; // al
  char *v20; // rdx
  _BYTE *v21; // rax
  _QWORD *v22; // r13
  __int64 v23; // r12
  __int64 *v24; // r14
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 *v27; // r13
  __int64 *i; // r13
  _QWORD *v29; // rax
  __int64 v30; // rax
  __int64 *v31; // rcx
  _QWORD *v32; // rax
  __int64 v33; // rax
  __int64 v34; // r14
  _QWORD *v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rsi
  __int64 v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rsi
  _QWORD *v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rcx
  int v44; // eax
  int v45; // r11d
  int v46; // eax
  unsigned int v47; // esi
  int v48; // eax
  __int64 v49; // rax
  __int64 *v52; // [rsp+28h] [rbp-88h]
  __int64 *v53; // [rsp+30h] [rbp-80h]
  char *v54; // [rsp+38h] [rbp-78h]
  __int64 *v55; // [rsp+38h] [rbp-78h]
  __int64 v56; // [rsp+40h] [rbp-70h] BYREF
  char *v57; // [rsp+48h] [rbp-68h] BYREF
  _BYTE *v58; // [rsp+50h] [rbp-60h] BYREF
  __int64 v59; // [rsp+58h] [rbp-58h]
  _BYTE v60[80]; // [rsp+60h] [rbp-50h] BYREF

  v4 = *(__int64 **)a1;
  if ( !*(_QWORD *)a1 )
    return;
  v58 = v60;
  v59 = 0x400000000LL;
  sub_13F9EC0(a1, (__int64)&v58);
  v7 = &v58[8 * (unsigned int)v59];
  if ( v58 == v7 )
  {
    v56 = a2;
    goto LABEL_104;
  }
  v8 = *(_DWORD *)(a4 + 24);
  v9 = *(_QWORD *)(a4 + 8);
  v10 = v58;
  v11 = 0;
  v12 = v8 - 1;
  do
  {
    if ( v8 )
    {
      v13 = v12 & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
      v14 = (__int64 *)(v9 + 16LL * v13);
      v15 = *v14;
      if ( *v10 == *v14 )
      {
LABEL_8:
        v16 = (__int64 *)v14[1];
        if ( v16 )
        {
          if ( !v11 )
            goto LABEL_4;
          if ( v11 != v16 )
          {
            v17 = (__int64 *)v14[1];
            while ( 1 )
            {
              v17 = (__int64 *)*v17;
              if ( v11 == v17 )
                break;
              if ( !v17 )
                goto LABEL_5;
            }
LABEL_4:
            v11 = v16;
          }
        }
      }
      else
      {
        v44 = 1;
        while ( v15 != -8 )
        {
          v45 = v44 + 1;
          v13 = v12 & (v44 + v13);
          v14 = (__int64 *)(v9 + 16LL * v13);
          v15 = *v14;
          if ( *v10 == *v14 )
            goto LABEL_8;
          v44 = v45;
        }
      }
    }
LABEL_5:
    ++v10;
  }
  while ( v10 != (_QWORD *)v7 );
  v52 = v11;
  if ( v11 == v4 )
  {
    if ( v58 != v60 )
      _libc_free((unsigned __int64)v58);
    return;
  }
  v56 = a2;
  if ( !v11 )
  {
LABEL_104:
    v52 = 0;
    if ( (unsigned __int8)sub_13FD8B0(a4, &v56, &v57) )
    {
      *(_QWORD *)v57 = -16;
      --*(_DWORD *)(a4 + 16);
      ++*(_DWORD *)(a4 + 20);
    }
    goto LABEL_23;
  }
  v19 = sub_13FD8B0(a4, &v56, &v57);
  v20 = v57;
  if ( v19 )
    goto LABEL_22;
  v46 = *(_DWORD *)(a4 + 16);
  v47 = *(_DWORD *)(a4 + 24);
  ++*(_QWORD *)a4;
  v48 = v46 + 1;
  if ( 4 * v48 >= 3 * v47 )
  {
    v47 *= 2;
    goto LABEL_107;
  }
  if ( v47 - *(_DWORD *)(a4 + 20) - v48 <= v47 >> 3 )
  {
LABEL_107:
    sub_1400170(a4, v47);
    sub_13FD8B0(a4, &v56, &v57);
    v20 = v57;
    v48 = *(_DWORD *)(a4 + 16) + 1;
  }
  *(_DWORD *)(a4 + 16) = v48;
  if ( *(_QWORD *)v20 != -8 )
    --*(_DWORD *)(a4 + 20);
  v49 = v56;
  *((_QWORD *)v20 + 1) = 0;
  *(_QWORD *)v20 = v49;
LABEL_22:
  *((_QWORD *)v20 + 1) = v52;
LABEL_23:
  v57 = (char *)a1;
  v21 = sub_1A4EE90((_QWORD *)v4[1], v4[2], (__int64 *)&v57);
  v22 = *(_QWORD **)v21;
  sub_13FDAF0((__int64)(v4 + 1), v21);
  *v22 = 0;
  if ( v52 )
  {
    *(_QWORD *)a1 = v52;
    v57 = (char *)a1;
    sub_1A541E0((__int64)(v52 + 1), &v57);
  }
  else
  {
    v57 = (char *)a1;
    sub_1A541E0(a4 + 32, &v57);
  }
  v18 = v4;
  v23 = a1 + 56;
  while ( 2 )
  {
    v24 = (__int64 *)v18[4];
    v54 = (char *)v18[5];
    v25 = (v54 - (char *)v24) >> 5;
    v26 = (v54 - (char *)v24) >> 3;
    if ( v25 <= 0 )
    {
LABEL_64:
      if ( v26 != 2 )
      {
        if ( v26 != 3 )
        {
          if ( v26 != 1 )
          {
LABEL_67:
            v24 = (__int64 *)v54;
            goto LABEL_36;
          }
LABEL_94:
          if ( a2 == *v24 || sub_1377F70(v23, *v24) )
            goto LABEL_30;
          goto LABEL_67;
        }
        if ( a2 == *v24 || sub_1377F70(v23, *v24) )
          goto LABEL_30;
        ++v24;
      }
      if ( a2 == *v24 || sub_1377F70(v23, *v24) )
        goto LABEL_30;
      ++v24;
      goto LABEL_94;
    }
    v27 = &v24[4 * v25];
    while ( a2 != *v24 && !sub_1377F70(v23, *v24) )
    {
      v36 = v24[1];
      if ( a2 == v36 || sub_1377F70(v23, v36) )
      {
        ++v24;
        break;
      }
      v37 = v24[2];
      if ( a2 == v37 || sub_1377F70(v23, v37) )
      {
        v24 += 2;
        break;
      }
      v38 = v24[3];
      if ( a2 == v38 || sub_1377F70(v23, v38) )
      {
        v24 += 3;
        break;
      }
      v24 += 4;
      if ( v27 == v24 )
      {
        v26 = (v54 - (char *)v24) >> 3;
        goto LABEL_64;
      }
    }
LABEL_30:
    if ( v54 != (char *)v24 )
    {
      for ( i = v24 + 1; v54 != (char *)i; ++i )
      {
        if ( a2 != *i && !sub_1377F70(v23, *i) )
          *v24++ = *i;
      }
    }
LABEL_36:
    sub_13E5810((__int64)(v18 + 4), (char *)v24, v54);
    v29 = (_QWORD *)v18[8];
    if ( (_QWORD *)v18[9] == v29 )
    {
      v41 = &v29[*((unsigned int *)v18 + 21)];
      if ( v29 == v41 )
      {
LABEL_82:
        v29 = v41;
      }
      else
      {
        while ( a2 != *v29 )
        {
          if ( v41 == ++v29 )
            goto LABEL_82;
        }
      }
    }
    else
    {
      v29 = sub_16CC9F0((__int64)(v18 + 7), a2);
      if ( a2 == *v29 )
      {
        v42 = v18[9];
        if ( v42 == v18[8] )
          v43 = *((unsigned int *)v18 + 21);
        else
          v43 = *((unsigned int *)v18 + 20);
        v41 = (_QWORD *)(v42 + 8 * v43);
      }
      else
      {
        v30 = v18[9];
        if ( v30 != v18[8] )
          goto LABEL_39;
        v29 = (_QWORD *)(v30 + 8LL * *((unsigned int *)v18 + 21));
        v41 = v29;
      }
    }
    if ( v29 != v41 )
    {
      *v29 = -2;
      ++*((_DWORD *)v18 + 22);
    }
LABEL_39:
    v31 = *(__int64 **)(a1 + 32);
    v55 = *(__int64 **)(a1 + 40);
    if ( v31 != v55 )
    {
      while ( 1 )
      {
        v34 = *v31;
        v32 = (_QWORD *)v18[8];
        if ( (_QWORD *)v18[9] == v32 )
        {
          v35 = &v32[*((unsigned int *)v18 + 21)];
          if ( v32 == v35 )
          {
LABEL_71:
            v32 = v35;
          }
          else
          {
            while ( v34 != *v32 )
            {
              if ( v35 == ++v32 )
                goto LABEL_71;
            }
          }
          goto LABEL_49;
        }
        v53 = v31;
        v32 = sub_16CC9F0((__int64)(v18 + 7), *v31);
        v31 = v53;
        if ( v34 == *v32 )
          break;
        v33 = v18[9];
        if ( v33 == v18[8] )
        {
          v32 = (_QWORD *)(v33 + 8LL * *((unsigned int *)v18 + 21));
          v35 = v32;
LABEL_49:
          if ( v32 != v35 )
          {
            *v32 = -2;
            ++*((_DWORD *)v18 + 22);
          }
        }
        if ( v55 == ++v31 )
          goto LABEL_15;
      }
      v39 = v18[9];
      if ( v39 == v18[8] )
        v40 = *((unsigned int *)v18 + 21);
      else
        v40 = *((unsigned int *)v18 + 20);
      v35 = (_QWORD *)(v39 + 8 * v40);
      goto LABEL_49;
    }
LABEL_15:
    sub_1AE4880(v18, a3, a4, 0);
    v18 = (__int64 *)*v18;
    if ( v18 != v52 )
      continue;
    break;
  }
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
}
