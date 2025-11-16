// Function: sub_2C295B0
// Address: 0x2c295b0
//
__int64 __fastcall sub_2C295B0(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 v6; // rax
  __int64 *v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r15
  __int64 i; // rbx
  __int64 v23; // rax
  __int64 v24; // r8
  unsigned __int64 v25; // rdx
  _BYTE *v26; // rdi
  unsigned __int64 v27; // r8
  const void *v28; // r11
  size_t v29; // r14
  int v30; // r15d
  __int64 *v31; // rdi
  __int64 *v32; // r14
  __int64 *v33; // r15
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // r8
  __int64 v37; // rax
  const void *v39; // [rsp+20h] [rbp-D0h]
  _BYTE *v40; // [rsp+28h] [rbp-C8h]
  _BYTE *v41; // [rsp+38h] [rbp-B8h]
  _BYTE *v42; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v43; // [rsp+48h] [rbp-A8h]
  _BYTE v44[48]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 *v45; // [rsp+80h] [rbp-70h] BYREF
  __int64 v46; // [rsp+88h] [rbp-68h]
  _BYTE v47[96]; // [rsp+90h] [rbp-60h] BYREF

  v42 = v44;
  v43 = 0x600000000LL;
  v4 = sub_2AAFF80((__int64)a2);
  v5 = *(__int64 **)(v4 + 112);
  v6 = 8LL * *(unsigned int *)(v4 + 120);
  v7 = &v5[(unsigned __int64)v6 / 8];
  v8 = v6 >> 3;
  v9 = v6 >> 5;
  if ( v9 )
  {
    v10 = &v5[4 * v9];
    while ( *(_BYTE *)(*v5 - 32) != 15 )
    {
      if ( *(_BYTE *)(v5[1] - 32) == 15 )
      {
        ++v5;
        goto LABEL_13;
      }
      if ( *(_BYTE *)(v5[2] - 32) == 15 )
      {
        v5 += 2;
        goto LABEL_13;
      }
      if ( *(_BYTE *)(v5[3] - 32) == 15 )
      {
        v5 += 3;
        goto LABEL_13;
      }
      v5 += 4;
      if ( v10 == v5 )
      {
        v8 = v7 - v5;
        goto LABEL_9;
      }
    }
    goto LABEL_13;
  }
LABEL_9:
  switch ( v8 )
  {
    case 2LL:
LABEL_53:
      if ( *(_BYTE *)(*v5 - 32) == 15 )
        goto LABEL_13;
      ++v5;
LABEL_55:
      if ( *(_BYTE *)(*v5 - 32) != 15 )
        v5 = v7;
      goto LABEL_13;
    case 3LL:
      if ( *(_BYTE *)(*v5 - 32) == 15 )
        goto LABEL_13;
      ++v5;
      goto LABEL_53;
    case 1LL:
      goto LABEL_55;
  }
  v5 = v7;
LABEL_13:
  v11 = sub_2AAFF80((__int64)a2);
  v15 = *(unsigned int *)(v11 + 120);
  if ( v5 != (__int64 *)(*(_QWORD *)(v11 + 112) + 8 * v15) )
  {
    v16 = *v5;
    if ( *v5 )
      v16 = *v5 + 56;
    sub_2AB9420((__int64)&v42, v16, v15, v12, v13, v14);
  }
  v17 = sub_2BF3F10(a2);
  v18 = sub_2BF04D0(v17);
  v19 = sub_2BF05A0(v18);
  v21 = *(_QWORD *)(v18 + 120);
  for ( i = v19; v21 != i; v21 = *(_QWORD *)(v21 + 8) )
  {
    while ( 1 )
    {
      if ( !v21 )
        BUG();
      if ( *(_BYTE *)(v21 - 16) == 33 && sub_2C1B260(v21 - 24) )
        break;
      v21 = *(_QWORD *)(v21 + 8);
      if ( v21 == i )
        goto LABEL_26;
    }
    v23 = (unsigned int)v43;
    v24 = v21 + 72;
    v25 = (unsigned int)v43 + 1LL;
    if ( v25 > HIDWORD(v43) )
    {
      sub_C8D5F0((__int64)&v42, v44, v25, 8u, v24, v20);
      v23 = (unsigned int)v43;
      v24 = v21 + 72;
    }
    *(_QWORD *)&v42[8 * v23] = v24;
    LODWORD(v43) = v43 + 1;
  }
LABEL_26:
  v26 = v42;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  v40 = &v26[8 * (unsigned int)v43];
  if ( v40 == v26 )
    goto LABEL_43;
  v41 = v26;
  do
  {
    v27 = *(unsigned int *)(*(_QWORD *)v41 + 24LL);
    v28 = *(const void **)(*(_QWORD *)v41 + 16LL);
    v29 = 8 * v27;
    v30 = *(_DWORD *)(*(_QWORD *)v41 + 24LL);
    v45 = (__int64 *)v47;
    v46 = 0x600000000LL;
    if ( v27 > 6 )
    {
      v39 = v28;
      sub_C8D5F0((__int64)&v45, v47, v27, 8u, v27, v20);
      v28 = v39;
      v31 = &v45[(unsigned int)v46];
    }
    else
    {
      v31 = (__int64 *)v47;
      if ( !v29 )
        goto LABEL_30;
    }
    memcpy(v31, v28, v29);
    v31 = v45;
    LODWORD(v29) = v46;
LABEL_30:
    LODWORD(v46) = v30 + v29;
    v32 = &v31[(unsigned int)(v30 + v29)];
    if ( v32 != v31 )
    {
      v33 = v31;
      do
      {
        while ( 1 )
        {
          v34 = *v33;
          if ( *(_BYTE *)(*v33 - 32) == 4 )
          {
            v35 = v34 + 56;
            if ( (unsigned __int8)sub_2C47490(v34 + 56, a2) )
              break;
          }
          if ( v32 == ++v33 )
            goto LABEL_38;
        }
        v37 = *(unsigned int *)(a1 + 8);
        if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v37 + 1, 8u, v36, v20);
          v37 = *(unsigned int *)(a1 + 8);
        }
        ++v33;
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v37) = v35;
        ++*(_DWORD *)(a1 + 8);
      }
      while ( v32 != v33 );
LABEL_38:
      v31 = v45;
    }
    if ( v31 != (__int64 *)v47 )
      _libc_free((unsigned __int64)v31);
    v41 += 8;
  }
  while ( v40 != v41 );
  v26 = v42;
LABEL_43:
  if ( v26 != v44 )
    _libc_free((unsigned __int64)v26);
  return a1;
}
