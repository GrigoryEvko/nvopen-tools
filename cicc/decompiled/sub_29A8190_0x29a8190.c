// Function: sub_29A8190
// Address: 0x29a8190
//
__int64 __fastcall sub_29A8190(__int64 *a1)
{
  __int64 v1; // rax
  __int64 *v2; // rax
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // r12d
  unsigned __int64 v17; // r15
  __int64 v18; // r14
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 *v21; // rdx
  __int64 v22; // r12
  __int64 v23; // rax
  _QWORD *v24; // rax
  bool v25; // si
  _BYTE **v26; // r13
  _BYTE **v27; // r14
  _BYTE *v28; // r12
  __int64 v30; // [rsp+18h] [rbp-168h]
  _BYTE *v31; // [rsp+18h] [rbp-168h]
  __m128i v32; // [rsp+20h] [rbp-160h] BYREF
  _BYTE *v33; // [rsp+30h] [rbp-150h]
  _BOOL4 v34; // [rsp+38h] [rbp-148h]
  __int64 v35; // [rsp+40h] [rbp-140h] BYREF
  __int64 *v36; // [rsp+48h] [rbp-138h]
  __int64 v37; // [rsp+50h] [rbp-130h]
  int v38; // [rsp+58h] [rbp-128h]
  char v39; // [rsp+5Ch] [rbp-124h]
  __int64 v40; // [rsp+60h] [rbp-120h] BYREF
  _BYTE *v41; // [rsp+68h] [rbp-118h]
  _BYTE *v42; // [rsp+70h] [rbp-110h]
  __int64 v43; // [rsp+78h] [rbp-108h]
  _BYTE v44[32]; // [rsp+80h] [rbp-100h] BYREF
  __int64 v45; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v46; // [rsp+A8h] [rbp-D8h]
  __int64 v47; // [rsp+B0h] [rbp-D0h]
  __int64 v48; // [rsp+B8h] [rbp-C8h]
  _BYTE *v49; // [rsp+C0h] [rbp-C0h]
  __int64 v50; // [rsp+C8h] [rbp-B8h]
  _BYTE v51[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v49 = v51;
  v50 = 0x400000000LL;
  v1 = *a1;
  v45 = 0;
  v46 = 0;
  v2 = *(__int64 **)(v1 + 32);
  v47 = 0;
  v48 = 0;
  v3 = sub_AA5930(*v2);
  v30 = v6;
  if ( v3 != v6 )
  {
    v7 = v3;
    do
    {
      v8 = a1[1];
      v9 = *a1;
      v35 = 6;
      v36 = 0;
      v37 = 0;
      v38 = 0;
      v40 = 0;
      v41 = 0;
      v42 = v44;
      v43 = 0x200000000LL;
      if ( !(unsigned __int8)sub_10238A0(v7, v9, v8, (__int64)&v35, 0, 0) )
        goto LABEL_13;
      v32.m128i_i64[0] = 0;
      v32.m128i_i64[1] = v7;
      v33 = v41;
      v34 = 0;
      if ( !v41 || (*v41 & 0xFB) != 0x2A && *v41 != 44 )
        goto LABEL_13;
      v34 = *((_QWORD *)v41 - 8) == v7;
      v10 = *(_QWORD *)&v41[32 * v34 - 64];
      if ( *(_BYTE *)v10 > 0x1Cu )
      {
        v11 = *a1;
        v12 = *(_QWORD *)(v10 + 40);
        if ( *(_BYTE *)(*a1 + 84) )
        {
          v13 = *(_QWORD **)(v11 + 64);
          v14 = &v13[*(unsigned int *)(v11 + 76)];
          if ( v13 != v14 )
          {
            while ( v12 != *v13 )
            {
              if ( v14 == ++v13 )
                goto LABEL_47;
            }
LABEL_13:
            if ( v42 != v44 )
              _libc_free((unsigned __int64)v42);
            if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
              sub_BD60C0(&v35);
            if ( !v7 )
              BUG();
            goto LABEL_19;
          }
        }
        else if ( sub_C8CA60(v11 + 56, v12) )
        {
          goto LABEL_13;
        }
      }
LABEL_47:
      sub_29A7E90(*(_QWORD *)(v7 + 16), 0, &v32, (__int64)&v45);
      sub_29A7E90(*((_QWORD *)v33 + 2), 0, &v32, (__int64)&v45);
      if ( v42 != v44 )
        _libc_free((unsigned __int64)v42);
      if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
        sub_BD60C0(&v35);
LABEL_19:
      v15 = *(_QWORD *)(v7 + 32);
      if ( !v15 )
        BUG();
      v7 = 0;
      if ( *(_BYTE *)(v15 - 24) == 84 )
        v7 = v15 - 24;
    }
    while ( v30 != v7 );
  }
  v16 = 0;
  if ( (_DWORD)v50 )
  {
    v17 = (unsigned __int64)v49;
    v35 = 0;
    v36 = &v40;
    v39 = 1;
    v31 = &v49[32 * (unsigned int)v50];
    v37 = 8;
    v38 = 0;
    v18 = *((_QWORD *)v49 + 1);
LABEL_25:
    v19 = v36;
    v20 = HIDWORD(v37);
    v21 = &v36[HIDWORD(v37)];
    if ( v36 == v21 )
    {
LABEL_52:
      if ( HIDWORD(v37) >= (unsigned int)v37 )
        goto LABEL_31;
      ++HIDWORD(v37);
      *v21 = v18;
      ++v35;
LABEL_32:
      v22 = *(_QWORD *)(v17 + 16);
      if ( !sub_98ED60((unsigned __int8 *)v22, 0, v22, a1[2], 0) )
      {
        sub_B44F30((unsigned __int8 *)v22);
        sub_DAC8D0(a1[1], (_BYTE *)v22);
      }
      if ( (*(_BYTE *)(v22 + 7) & 0x40) != 0 )
        v23 = *(_QWORD *)(v22 - 8);
      else
        v23 = v22 - 32LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF);
      sub_29A7000(a1, v23 + 32LL * *(unsigned int *)(v17 + 24));
      v24 = *(_QWORD **)(v18 - 8);
      v25 = *v24 == v22;
      if ( (*(_BYTE *)(v18 + 7) & 0x40) == 0 )
        v24 = (_QWORD *)(v18 - 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF));
      v17 += 32LL;
      sub_29A7000(a1, (__int64)&v24[4 * v25]);
      if ( v31 != (_BYTE *)v17 )
        goto LABEL_30;
    }
    else
    {
      while ( v18 != *v19 )
      {
        if ( v21 == ++v19 )
          goto LABEL_52;
      }
      while ( 1 )
      {
        v17 += 32LL;
        if ( v31 == (_BYTE *)v17 )
          break;
LABEL_30:
        v18 = *(_QWORD *)(v17 + 8);
        if ( v39 )
          goto LABEL_25;
LABEL_31:
        sub_C8CC70((__int64)&v35, v18, (__int64)v21, v20, v4, v5);
        if ( (_BYTE)v21 )
          goto LABEL_32;
      }
    }
    v26 = (_BYTE **)v49;
    v27 = (_BYTE **)&v49[32 * (unsigned int)v50];
    if ( v49 != (_BYTE *)v27 )
    {
      do
      {
        v28 = *v26;
        v26 += 4;
        sub_DAC8D0(a1[1], v28);
        sub_BD84D0((__int64)v28, *((_QWORD *)v28 - 4));
        sub_B43D60(v28);
      }
      while ( v27 != v26 );
    }
    v16 = 1;
    if ( !v39 )
      _libc_free((unsigned __int64)v36);
  }
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  sub_C7D6A0(v46, 32LL * (unsigned int)v48, 8);
  return v16;
}
