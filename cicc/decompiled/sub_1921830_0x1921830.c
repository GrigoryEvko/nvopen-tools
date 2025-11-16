// Function: sub_1921830
// Address: 0x1921830
//
__int64 __fastcall sub_1921830(__int64 *a1, int *a2, unsigned int *a3)
{
  __int64 v5; // rcx
  __int64 v6; // r15
  int v7; // esi
  int v8; // edx
  int v10; // edi
  int v11; // r8d
  __int64 v12; // r9
  int v13; // r13d
  unsigned __int64 v14; // r10
  unsigned __int64 v15; // r10
  unsigned int i; // eax
  _DWORD *v17; // r10
  unsigned int v18; // eax
  __int64 v19; // r13
  _BYTE *v20; // rdi
  __int64 v21; // rsi
  unsigned __int8 v22; // al
  unsigned int v23; // r12d
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // rdx
  int v27; // r11d
  unsigned int v28; // esi
  unsigned __int64 v29; // r9
  unsigned __int64 v30; // r9
  unsigned int j; // eax
  __int64 v32; // r9
  unsigned int v33; // eax
  __int64 v34; // rdx
  unsigned __int8 v35; // al
  _QWORD *v36; // rdi
  int v38; // eax
  __int64 v39; // rdx
  int v40; // r8d
  int v41; // r8d
  __int64 v42; // r10
  unsigned int v43; // r9d
  __int64 *v44; // rax
  __int64 v45; // r12
  int v46; // eax
  int v47; // eax
  int v48; // r8d
  __int64 v49; // rsi
  unsigned int v50; // ecx
  __int64 *v51; // rax
  __int64 v52; // r9
  int v53; // eax
  int v54; // eax
  int v55; // r10d
  int v56; // eax
  int v57; // r11d
  _BYTE *v58; // [rsp+0h] [rbp-90h] BYREF
  __int64 v59; // [rsp+8h] [rbp-88h]
  _BYTE v60[32]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v61; // [rsp+30h] [rbp-60h] BYREF
  __int64 v62; // [rsp+38h] [rbp-58h]
  _QWORD v63[10]; // [rsp+40h] [rbp-50h] BYREF

  v5 = a1[1];
  v6 = *a1;
  v7 = *(_DWORD *)(v5 + 24);
  v8 = v7;
  if ( v7 )
  {
    v10 = *a2;
    v11 = a2[1];
    v12 = *(_QWORD *)(v5 + 8);
    v13 = 1;
    v14 = ((((unsigned int)(37 * v11) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))
          - 1
          - ((unsigned __int64)(unsigned int)(37 * v11) << 32)) >> 22)
        ^ (((unsigned int)(37 * v11) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))
         - 1
         - ((unsigned __int64)(unsigned int)(37 * v11) << 32));
    v15 = ((9 * (((v14 - 1 - (v14 << 13)) >> 8) ^ (v14 - 1 - (v14 << 13)))) >> 15)
        ^ (9 * (((v14 - 1 - (v14 << 13)) >> 8) ^ (v14 - 1 - (v14 << 13))));
    for ( i = (v7 - 1) & (((v15 - 1 - (v15 << 27)) >> 31) ^ (v15 - 1 - ((_DWORD)v15 << 27))); ; i = (v7 - 1) & v18 )
    {
      v17 = (_DWORD *)(v12 + 56LL * i);
      if ( *v17 == v10 && v17[1] == v11 )
        break;
      if ( *v17 == -1 && v17[1] == -1 )
        goto LABEL_7;
      v18 = v13 + i;
      ++v13;
    }
    v39 = (unsigned int)v17[4];
    v58 = v60;
    v59 = 0x400000000LL;
    if ( (_DWORD)v39 )
    {
      sub_191FF30((__int64)&v58, (__int64)(v17 + 2), v39, v5, v11, v12);
      v5 = a1[1];
      v19 = *a1;
      v20 = v58;
      v8 = *(_DWORD *)(v5 + 24);
    }
    else
    {
      v8 = v7;
      v19 = v6;
      v20 = v60;
    }
  }
  else
  {
LABEL_7:
    v19 = v6;
    v58 = v60;
    v20 = v60;
    v59 = 0x400000000LL;
  }
  v21 = *(_QWORD *)v20;
  v22 = *(_BYTE *)(*(_QWORD *)v20 + 16LL);
  if ( v22 == 5 )
  {
    v23 = 2;
  }
  else
  {
    v23 = 1;
    if ( v22 != 9 )
    {
      v23 = 0;
      if ( v22 > 0x10u )
      {
        if ( v22 == 17 )
        {
          v23 = *(_DWORD *)(v21 + 32) + 3;
          goto LABEL_13;
        }
        v40 = *(_DWORD *)(v6 + 288);
        v23 = -1;
        if ( v40 )
        {
          v41 = v40 - 1;
          v42 = *(_QWORD *)(v6 + 272);
          v43 = v41 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v44 = (__int64 *)(v42 + 16LL * v43);
          v45 = *v44;
          if ( v21 == *v44 )
          {
LABEL_40:
            v46 = *((_DWORD *)v44 + 2);
            if ( v46 )
            {
              v23 = v46 + *(_DWORD *)(v6 + 632) + 4;
              goto LABEL_13;
            }
          }
          else
          {
            v56 = 1;
            while ( v45 != -8 )
            {
              v57 = v56 + 1;
              v43 = v41 & (v56 + v43);
              v44 = (__int64 *)(v42 + 16LL * v43);
              v45 = *v44;
              if ( v21 == *v44 )
                goto LABEL_40;
              v56 = v57;
            }
          }
          v23 = -1;
        }
      }
    }
  }
LABEL_13:
  if ( !v8 )
  {
LABEL_19:
    v61 = v63;
    v62 = 0x400000000LL;
LABEL_20:
    v34 = v63[0];
    v35 = *(_BYTE *)(v63[0] + 16LL);
    if ( v35 == 5 )
    {
      LOBYTE(v6) = v23 <= 1;
      goto LABEL_29;
    }
    v36 = v63;
    goto LABEL_22;
  }
  v24 = *(_QWORD *)(v5 + 8);
  v25 = *a3;
  v26 = (unsigned int)(v8 - 1);
  v27 = 1;
  v28 = a3[1];
  v29 = ((((37 * v28) | ((unsigned __int64)(unsigned int)(37 * v25) << 32)) - 1 - ((unsigned __int64)(37 * v28) << 32)) >> 22)
      ^ (((37 * v28) | ((unsigned __int64)(unsigned int)(37 * v25) << 32)) - 1 - ((unsigned __int64)(37 * v28) << 32));
  v30 = ((9 * (((v29 - 1 - (v29 << 13)) >> 8) ^ (v29 - 1 - (v29 << 13)))) >> 15)
      ^ (9 * (((v29 - 1 - (v29 << 13)) >> 8) ^ (v29 - 1 - (v29 << 13))));
  for ( j = v26 & (((v30 - 1 - (v30 << 27)) >> 31) ^ (v30 - 1 - ((_DWORD)v30 << 27))); ; j = v26 & v33 )
  {
    v32 = v24 + 56LL * j;
    if ( *(_QWORD *)v32 == __PAIR64__(v28, v25) )
      break;
    if ( *(_DWORD *)v32 == -1 && *(_DWORD *)(v32 + 4) == -1 )
      goto LABEL_19;
    v33 = v27 + j;
    ++v27;
  }
  v62 = 0x400000000LL;
  v38 = *(_DWORD *)(v32 + 16);
  v61 = v63;
  if ( !v38 )
    goto LABEL_20;
  sub_191FF30((__int64)&v61, v32 + 8, v26, v25, (int)&v61, v32);
  v36 = v61;
  LOBYTE(v6) = v23 <= 1;
  v34 = *v61;
  v35 = *(_BYTE *)(*v61 + 16LL);
  if ( v35 == 5 )
    goto LABEL_26;
LABEL_22:
  LOBYTE(v6) = v23 == 0;
  if ( v35 == 9 )
    goto LABEL_26;
  LODWORD(v6) = 0;
  if ( v35 <= 0x10u )
    goto LABEL_26;
  if ( v35 != 17 )
  {
    v47 = *(_DWORD *)(v19 + 288);
    if ( v47 )
    {
      v48 = v47 - 1;
      v49 = *(_QWORD *)(v19 + 272);
      v50 = (v47 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v51 = (__int64 *)(v49 + 16LL * v50);
      v52 = *v51;
      if ( *v51 == v34 )
      {
LABEL_44:
        v53 = *((_DWORD *)v51 + 2);
        if ( v53 )
        {
          LOBYTE(v6) = v23 < v53 + *(_DWORD *)(v19 + 632) + 4;
          goto LABEL_26;
        }
      }
      else
      {
        v54 = 1;
        while ( v52 != -8 )
        {
          v55 = v54 + 1;
          v50 = v48 & (v54 + v50);
          v51 = (__int64 *)(v49 + 16LL * v50);
          v52 = *v51;
          if ( v34 == *v51 )
            goto LABEL_44;
          v54 = v55;
        }
      }
    }
    LOBYTE(v6) = v23 != -1;
    goto LABEL_26;
  }
  LOBYTE(v6) = *(_DWORD *)(v34 + 32) + 3 > v23;
LABEL_26:
  if ( v36 != v63 )
    _libc_free((unsigned __int64)v36);
  v20 = v58;
LABEL_29:
  if ( v20 != v60 )
    _libc_free((unsigned __int64)v20);
  return (unsigned int)v6;
}
