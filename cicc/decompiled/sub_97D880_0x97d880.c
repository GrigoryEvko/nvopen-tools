// Function: sub_97D880
// Address: 0x97d880
//
__int64 __fastcall sub_97D880(__int64 a1, _BYTE *a2, __int64 *a3)
{
  char v6; // cl
  __int64 v7; // rdx
  __int64 *v8; // r15
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 *v11; // rbx
  __int64 v12; // r13
  __int64 **v13; // rcx
  __int64 v14; // r14
  __int64 v15; // rdi
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  char v20; // si
  __int64 v21; // rax
  int v22; // esi
  char *v24; // r11
  char *v25; // rbx
  __int64 v26; // rcx
  __int64 v27; // rdx
  char *v28; // rax
  char *v29; // rdx
  __int64 **v30; // rax
  __int64 *v31; // r15
  _BYTE *v32; // r9
  char v33; // al
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned int v37; // eax
  __int64 *v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // [rsp+0h] [rbp-F0h]
  __int64 *v41; // [rsp+0h] [rbp-F0h]
  __int64 **v42; // [rsp+8h] [rbp-E8h]
  __int64 *v43; // [rsp+10h] [rbp-E0h]
  __int64 *v44; // [rsp+10h] [rbp-E0h]
  _BYTE *v45; // [rsp+10h] [rbp-E0h]
  _BYTE *v46; // [rsp+18h] [rbp-D8h]
  _BYTE *v47; // [rsp+18h] [rbp-D8h]
  _BYTE *v48; // [rsp+18h] [rbp-D8h]
  __int64 v49; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v50; // [rsp+28h] [rbp-C8h]
  __int64 v51; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v52; // [rsp+38h] [rbp-B8h]
  __int64 *v53; // [rsp+70h] [rbp-80h] BYREF
  __int64 v54; // [rsp+78h] [rbp-78h]
  __int64 v55; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v56; // [rsp+88h] [rbp-68h]
  char v57; // [rsp+C0h] [rbp-30h] BYREF

  v6 = *(_BYTE *)(a1 + 7) & 0x40;
  v7 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_BYTE *)a1 != 84 )
  {
    v24 = (char *)(a1 - v7);
    if ( v6 )
      v24 = *(char **)(a1 - 8);
    v25 = &v24[v7];
    v26 = v7 >> 5;
    v27 = v7 >> 7;
    if ( v27 )
    {
      v28 = v24;
      v29 = &v24[128 * v27];
      while ( **(_BYTE **)v28 <= 0x15u )
      {
        if ( **((_BYTE **)v28 + 4) > 0x15u )
        {
          v28 += 32;
          break;
        }
        if ( **((_BYTE **)v28 + 8) > 0x15u )
        {
          v28 += 64;
          break;
        }
        if ( **((_BYTE **)v28 + 12) > 0x15u )
        {
          v28 += 96;
          break;
        }
        v28 += 128;
        if ( v29 == v28 )
        {
          v26 = (v25 - v28) >> 5;
          goto LABEL_62;
        }
      }
LABEL_43:
      v14 = 0;
      if ( v25 != v28 )
        return v14;
      goto LABEL_44;
    }
    v28 = v24;
LABEL_62:
    if ( v26 != 2 )
    {
      if ( v26 != 3 )
      {
        if ( v26 != 1 )
          goto LABEL_44;
        goto LABEL_65;
      }
      if ( **(_BYTE **)v28 > 0x15u )
        goto LABEL_43;
      v28 += 32;
    }
    if ( **(_BYTE **)v28 > 0x15u )
      goto LABEL_43;
    v28 += 32;
LABEL_65:
    if ( **(_BYTE **)v28 > 0x15u )
      goto LABEL_43;
LABEL_44:
    v49 = 0;
    v30 = (__int64 **)&v51;
    v50 = 1;
    do
    {
      *v30 = (__int64 *)-4096LL;
      v30 += 2;
    }
    while ( v30 != &v53 );
    v53 = &v55;
    v54 = 0x800000000LL;
    if ( v25 == v24 )
    {
      v38 = &v55;
      v39 = 0;
    }
    else
    {
      v31 = (__int64 *)v24;
      do
      {
        v32 = (_BYTE *)*v31;
        v33 = *(_BYTE *)*v31;
        if ( v33 == 5 || v33 == 11 )
        {
          v44 = a3;
          v47 = a2;
          v34 = sub_97B040(*v31, (__int64)a2, (__int64)a3, (__int64)&v49);
          a3 = v44;
          a2 = v47;
          v32 = (_BYTE *)v34;
        }
        v35 = (unsigned int)v54;
        v36 = (unsigned int)v54 + 1LL;
        if ( v36 > HIDWORD(v54) )
        {
          v41 = a3;
          v45 = a2;
          v48 = v32;
          sub_C8D5F0(&v53, &v55, v36, 8);
          v35 = (unsigned int)v54;
          a3 = v41;
          a2 = v45;
          v32 = v48;
        }
        v31 += 4;
        v53[v35] = (__int64)v32;
        v37 = v54 + 1;
        LODWORD(v54) = v54 + 1;
      }
      while ( v25 != (char *)v31 );
      v38 = v53;
      v39 = v37;
    }
    v14 = sub_97D230((unsigned __int8 *)a1, v38, v39, a2, a3, 1u);
    if ( v53 != &v55 )
      _libc_free(v53, v38);
    if ( (v50 & 1) == 0 )
      sub_C7D6A0(v51, 16LL * v52, 8);
    return v14;
  }
  v53 = 0;
  v8 = (__int64 *)a1;
  v9 = &v55;
  v54 = 1;
  do
  {
    *v9 = -4096;
    v9 += 2;
  }
  while ( v9 != (__int64 *)&v57 );
  v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 72LL) + 80LL);
  if ( v10 )
    v10 -= 24;
  v11 = (__int64 *)(a1 - v7);
  if ( v6 )
  {
    v11 = *(__int64 **)(a1 - 8);
    v8 = &v11[(unsigned __int64)v7 / 8];
  }
  v12 = 0;
  v13 = &v53;
  v14 = 0;
  if ( v8 != v11 )
  {
    while ( 1 )
    {
      v15 = *v11;
      v16 = *(unsigned __int8 *)*v11;
      if ( (unsigned int)(v16 - 12) <= 1 )
        goto LABEL_10;
      if ( (unsigned __int8)v16 > 0x15u )
      {
        v18 = *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL * *(unsigned int *)(a1 + 72) + v12);
        if ( v10 == v18 )
          goto LABEL_30;
        v19 = *(_QWORD *)(v18 + 16);
        while ( v19 )
        {
          v20 = **(_BYTE **)(v19 + 24);
          v21 = v19;
          v19 = *(_QWORD *)(v19 + 8);
          if ( (unsigned __int8)(v20 - 30) <= 0xAu )
          {
            v22 = 0;
            while ( 1 )
            {
              v21 = *(_QWORD *)(v21 + 8);
              if ( !v21 )
                break;
              while ( (unsigned __int8)(**(_BYTE **)(v21 + 24) - 30) <= 0xAu )
              {
                v21 = *(_QWORD *)(v21 + 8);
                ++v22;
                if ( !v21 )
                  goto LABEL_29;
              }
            }
LABEL_29:
            if ( v22 == -1 )
              break;
            goto LABEL_30;
          }
        }
LABEL_10:
        v11 += 4;
        v12 += 8;
        if ( v8 == v11 )
          goto LABEL_19;
      }
      else
      {
        if ( (_BYTE)v16 == 5 || (_BYTE)v16 == 11 )
        {
          v40 = v10;
          v42 = v13;
          v43 = a3;
          v46 = a2;
          v17 = sub_97B040(v15, (__int64)a2, (__int64)a3, (__int64)v13);
          v10 = v40;
          v13 = v42;
          a3 = v43;
          a2 = v46;
          v15 = v17;
        }
        if ( v14 && v15 != v14 )
        {
LABEL_30:
          v14 = 0;
          goto LABEL_31;
        }
        v11 += 4;
        v14 = v15;
        v12 += 8;
        if ( v8 == v11 )
        {
LABEL_19:
          if ( v14 )
            goto LABEL_31;
          break;
        }
      }
    }
  }
  v14 = sub_ACA8A0(*(_QWORD *)(a1 + 8));
LABEL_31:
  if ( (v54 & 1) == 0 )
    sub_C7D6A0(v55, 16LL * v56, 8);
  return v14;
}
