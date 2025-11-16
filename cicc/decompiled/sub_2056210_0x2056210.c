// Function: sub_2056210
// Address: 0x2056210
//
void __fastcall sub_2056210(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8,
        unsigned int a9,
        unsigned __int8 a10)
{
  __int64 v10; // r15
  __int64 v11; // r14
  int v12; // r13d
  __int64 v13; // r12
  bool v14; // al
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  unsigned __int8 v18; // al
  int v19; // esi
  __int64 v20; // rax
  __int64 v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // r8
  unsigned __int64 v26; // rdi
  __int64 v27; // rsi
  unsigned int v28; // edi
  _QWORD *v29; // rsi
  __int64 v30; // r12
  __int64 v31; // rax
  unsigned int v32; // edi
  _QWORD *v33; // rsi
  __int64 v34; // r12
  int v35; // [rsp+8h] [rbp-88h]
  unsigned __int64 *v36; // [rsp+10h] [rbp-80h]
  __int64 v37; // [rsp+30h] [rbp-60h]
  int v38; // [rsp+30h] [rbp-60h]
  __int64 v39; // [rsp+30h] [rbp-60h]
  __int64 v40; // [rsp+38h] [rbp-58h]
  __int64 v41; // [rsp+38h] [rbp-58h]
  unsigned __int64 v42; // [rsp+38h] [rbp-58h]
  __int64 v43; // [rsp+38h] [rbp-58h]
  unsigned __int64 v44; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v45[14]; // [rsp+58h] [rbp-38h] BYREF

  while ( 1 )
  {
    v10 = a3;
    v11 = a4;
    v12 = a1;
    v13 = a2;
    v37 = a5;
    v40 = a6;
    v14 = sub_15FB730(a2, a2, a3, a4);
    a6 = v40;
    a5 = v37;
    if ( !v14 )
      break;
    v17 = *(_QWORD *)(a2 + 8);
    if ( !v17 )
      break;
    if ( *(_QWORD *)(v17 + 8) )
      break;
    v39 = v40;
    v43 = a5;
    v31 = sub_15FB7F0(a2, a2, v15, v16);
    a5 = v43;
    a6 = v39;
    a2 = v31;
    if ( *(_BYTE *)(v31 + 16) > 0x17u && *(_QWORD *)(v31 + 40) != *(_QWORD *)(v43 + 40) )
      break;
    a4 = v11;
    a10 ^= 1u;
    a3 = v10;
  }
  v18 = *(_BYTE *)(v13 + 16);
  if ( v18 <= 0x17u )
    goto LABEL_28;
  v19 = v18 - 24;
  if ( !a10 )
    goto LABEL_8;
  if ( v18 != 50 )
  {
    if ( v18 == 51 )
    {
      v19 = 26;
LABEL_10:
      if ( v19 == a7 )
        goto LABEL_11;
LABEL_28:
      sub_2055E10(a1, v13, v10, v11, a5, a6, a8, a9, a10);
      return;
    }
LABEL_8:
    if ( (unsigned int)v18 - 35 > 0x11 && (unsigned __int8)(v18 - 75) > 1u )
      goto LABEL_28;
    goto LABEL_10;
  }
  if ( a7 != 27 )
    goto LABEL_28;
LABEL_11:
  v20 = *(_QWORD *)(v13 + 8);
  if ( !v20 )
    goto LABEL_28;
  if ( *(_QWORD *)(v20 + 8) )
    goto LABEL_28;
  v21 = *(_QWORD *)(a5 + 40);
  if ( v21 != *(_QWORD *)(v13 + 40) )
    goto LABEL_28;
  v22 = (*(_BYTE *)(v13 + 23) & 0x40) != 0
      ? *(_QWORD **)(v13 - 8)
      : (_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(*v22 + 16LL) > 0x17u && v21 != *(_QWORD *)(*v22 + 40LL) )
    goto LABEL_28;
  v23 = v22[3];
  if ( *(_BYTE *)(v23 + 16) > 0x17u && v21 != *(_QWORD *)(v23 + 40) )
    goto LABEL_28;
  v35 = a6;
  v41 = a5;
  v24 = sub_1E0B6F0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL), v21);
  v25 = v41;
  v42 = (unsigned __int64)v24;
  v38 = v25;
  v36 = *(unsigned __int64 **)(v25 + 8);
  sub_1DD8DC0(*(_QWORD *)(v25 + 56) + 320LL, (__int64)v24);
  v26 = *v36;
  v27 = *(_QWORD *)v42;
  *(_QWORD *)(v42 + 8) = v36;
  v26 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v42 = v26 | v27 & 7;
  *(_QWORD *)(v26 + 8) = v42;
  *v36 = v42 | *v36 & 7;
  if ( a7 == 27 )
  {
    v32 = (a8 >> 1) + a9;
    if ( (a8 >> 1) + (unsigned __int64)a9 > 0x80000000 )
      v32 = 0x80000000;
    if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
      v33 = *(_QWORD **)(v13 - 8);
    else
      v33 = (_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
    sub_2056210(v12, *v33, v10, v42, v38, v35, 27, a8 >> 1, v32, a10);
    v44 = (a8 >> 1) | ((unsigned __int64)a9 << 32);
    sub_1953BB0((unsigned int *)&v44, v45);
    if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
      v34 = *(_QWORD *)(v13 - 8);
    else
      v34 = v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
    sub_2056210(v12, *(_QWORD *)(v34 + 24), v10, v11, v42, v35, 27, v44, HIDWORD(v44), a10);
  }
  else
  {
    v28 = 0x80000000;
    if ( a8 + (unsigned __int64)(a9 >> 1) <= 0x80000000 )
      v28 = a8 + (a9 >> 1);
    if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
      v29 = *(_QWORD **)(v13 - 8);
    else
      v29 = (_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
    sub_2056210(v12, *v29, v42, v11, v38, v35, a7, v28, a9 >> 1, a10);
    v44 = a8 | ((unsigned __int64)(a9 >> 1) << 32);
    sub_1953BB0((unsigned int *)&v44, v45);
    if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
      v30 = *(_QWORD *)(v13 - 8);
    else
      v30 = v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
    sub_2056210(v12, *(_QWORD *)(v30 + 24), v10, v11, v42, v35, a7, v44, HIDWORD(v44), a10);
  }
}
