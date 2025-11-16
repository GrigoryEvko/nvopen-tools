// Function: sub_2C1A170
// Address: 0x2c1a170
//
__int64 __fastcall sub_2C1A170(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned int v9; // r15d
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // r13
  unsigned __int8 *v13; // rax
  int v14; // edx
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // r15d
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  int v26; // r9d
  __int64 v27; // rdi
  __int64 v28; // r12
  __int64 v30; // rcx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  __int128 v34; // [rsp-18h] [rbp-1A8h]
  __int64 v35; // [rsp+8h] [rbp-188h]
  __int64 v36; // [rsp+18h] [rbp-178h]
  char *v38; // [rsp+40h] [rbp-150h] BYREF
  __int64 v39; // [rsp+48h] [rbp-148h]
  _BYTE v40[48]; // [rsp+50h] [rbp-140h] BYREF
  char *v41; // [rsp+80h] [rbp-110h] BYREF
  __int64 v42; // [rsp+88h] [rbp-108h]
  _BYTE v43[48]; // [rsp+90h] [rbp-100h] BYREF
  __int64 v44[3]; // [rsp+C0h] [rbp-D0h] BYREF
  char *v45; // [rsp+D8h] [rbp-B8h]
  char v46; // [rsp+E8h] [rbp-A8h] BYREF
  char *v47; // [rsp+108h] [rbp-88h]
  char v48; // [rsp+118h] [rbp-78h] BYREF

  v38 = v40;
  v7 = *(_QWORD *)(a1 + 48);
  v39 = 0x600000000LL;
  v8 = v7 + 8LL * *(unsigned int *)(a1 + 56);
  if ( v8 == v7 )
    goto LABEL_13;
  v9 = 0;
  while ( 1 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)v7 + 40LL);
    if ( !v12 )
      break;
LABEL_3:
    v10 = (unsigned int)v39;
    v11 = (unsigned int)v39 + 1LL;
    if ( v11 > HIDWORD(v39) )
    {
      sub_C8D5F0((__int64)&v38, v40, v11, 8u, a5, a6);
      v10 = (unsigned int)v39;
    }
    *(_QWORD *)&v38[8 * v10] = v12;
    LODWORD(v39) = v39 + 1;
LABEL_6:
    v7 += 8;
    ++v9;
    if ( v8 == v7 )
      goto LABEL_13;
  }
  if ( sub_B5A760(*(_DWORD *)(a1 + 160)) )
  {
    v31 = (unsigned int)v39;
    v32 = (unsigned int)v39 + 1LL;
    if ( v32 > HIDWORD(v39) )
    {
      sub_C8D5F0((__int64)&v38, v40, v32, 8u, a5, a6);
      v31 = (unsigned int)v39;
    }
    *(_QWORD *)&v38[8 * v31] = 0;
    LODWORD(v39) = v39 + 1;
    goto LABEL_6;
  }
  v13 = *(unsigned __int8 **)(a1 + 136);
  if ( v13 )
  {
    v14 = *v13;
    if ( (unsigned __int8)v14 > 0x1Cu )
    {
      v15 = (unsigned int)(v14 - 34);
      if ( (unsigned __int8)v15 <= 0x33u )
      {
        v30 = 0x8000000000041LL;
        if ( _bittest64(&v30, v15) )
        {
          v12 = *(_QWORD *)&v13[32 * (v9 - (unsigned __int64)(*((_DWORD *)v13 + 1) & 0x7FFFFFF))];
          goto LABEL_3;
        }
      }
    }
  }
  LODWORD(v39) = 0;
LABEL_13:
  v18 = sub_2BFD6A0((__int64)(a3 + 2), a1 + 96);
  v44[0] = a2;
  if ( *(_BYTE *)(v18 + 8) == 15 )
    v36 = (__int64)sub_E454C0(v18, a2, v16, v17, v19, v20);
  else
    v36 = sub_2AAEDF0(v18, a2);
  v41 = v43;
  v42 = 0x600000000LL;
  if ( *(_DWORD *)(a1 + 56) )
  {
    v21 = 0;
    do
    {
      v22 = sub_2BFD6A0((__int64)(a3 + 2), *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * v21));
      v44[0] = a2;
      if ( ((*(_BYTE *)(v22 + 8) - 7) & 0xFD) != 0 && (BYTE4(a2) || (_DWORD)a2 != 1) )
      {
        LODWORD(v44[0]) = a2;
        BYTE4(v44[0]) = BYTE4(a2);
        v22 = sub_BCE1B0((__int64 *)v22, v44[0]);
      }
      v24 = (unsigned int)v42;
      v25 = (unsigned int)v42 + 1LL;
      if ( v25 > HIDWORD(v42) )
      {
        v35 = v22;
        sub_C8D5F0((__int64)&v41, v43, v25, 8u, v22, v23);
        v24 = (unsigned int)v42;
        v22 = v35;
      }
      ++v21;
      *(_QWORD *)&v41[8 * v24] = v22;
      LODWORD(v42) = v42 + 1;
    }
    while ( v21 != *(_DWORD *)(a1 + 56) );
  }
  v26 = 0;
  if ( *(_BYTE *)(a1 + 152) == 5 )
    v26 = sub_2C1A110(a1);
  v27 = *(_QWORD *)(a1 + 136);
  if ( v27 )
  {
    if ( *(_BYTE *)v27 == 85
      && (v33 = *(_QWORD *)(v27 - 32)) != 0
      && !*(_BYTE *)v33
      && *(_QWORD *)(v33 + 24) == *(_QWORD *)(v27 + 80) )
    {
      if ( (*(_BYTE *)(v33 + 33) & 0x20) == 0 )
        v27 = 0;
    }
    else
    {
      v27 = 0;
    }
  }
  *((_QWORD *)&v34 + 1) = 1;
  *(_QWORD *)&v34 = 0;
  sub_DF8E30(
    (__int64)v44,
    *(_DWORD *)(a1 + 160),
    v36,
    v38,
    (unsigned int)v39,
    v26,
    v41,
    (unsigned int)v42,
    v27,
    v34,
    a3[1]);
  v28 = sub_DFD690(*a3, (__int64)v44);
  if ( v47 != &v48 )
    _libc_free((unsigned __int64)v47);
  if ( v45 != &v46 )
    _libc_free((unsigned __int64)v45);
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  return v28;
}
