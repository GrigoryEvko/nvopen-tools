// Function: sub_3447950
// Address: 0x3447950
//
__int64 __fastcall sub_3447950(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  unsigned int v7; // r14d
  __int64 v11; // rax
  __int16 v12; // cx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // r15d
  unsigned int v18; // eax
  unsigned int v19; // r9d
  unsigned int v20; // ebx
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 (*v24)(); // rax
  unsigned __int64 v25; // rdx
  __int64 (*v26)(); // rax
  unsigned int v27; // eax
  __int64 v28; // rbx
  int v29; // r9d
  __int64 v30; // rdx
  int v31; // r9d
  __int128 v32; // rax
  int v33; // r9d
  unsigned __int8 *v34; // rax
  int v35; // edx
  int v36; // [rsp+4h] [rbp-9Ch]
  __int16 v38; // [rsp+10h] [rbp-90h]
  int v39; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+10h] [rbp-90h]
  __int128 v41; // [rsp+10h] [rbp-90h]
  __int64 v43; // [rsp+28h] [rbp-78h]
  unsigned int v44; // [rsp+50h] [rbp-50h] BYREF
  __int64 v45; // [rsp+58h] [rbp-48h]
  __int64 v46; // [rsp+60h] [rbp-40h] BYREF
  int v47; // [rsp+68h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(a2 + 80);
  v45 = *(_QWORD *)(v11 + 8);
  v14 = *(_QWORD *)a6;
  LOWORD(v44) = v12;
  v43 = v14;
  v46 = v13;
  if ( v13 )
  {
    v38 = v12;
    sub_B96E90((__int64)&v46, v13, 1);
    v12 = v38;
  }
  v47 = *(_DWORD *)(a2 + 72);
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 17) <= 0xD3u )
      goto LABEL_6;
    v15 = *(_QWORD *)(a2 + 56);
    if ( !v15 )
      goto LABEL_6;
  }
  else
  {
    if ( sub_30070B0((__int64)&v44) )
      goto LABEL_6;
    v15 = *(_QWORD *)(a2 + 56);
    if ( !v15 )
      goto LABEL_6;
  }
  if ( *(_QWORD *)(v15 + 32) )
  {
LABEL_6:
    v16 = 0;
    goto LABEL_7;
  }
  if ( *(_DWORD *)(a5 + 8) <= 0x40u )
  {
    v19 = 1;
    if ( !*(_QWORD *)a5 )
      goto LABEL_17;
    _BitScanReverse64(&v25, *(_QWORD *)a5);
    v18 = 64 - (v25 ^ 0x3F);
  }
  else
  {
    v39 = *(_DWORD *)(a5 + 8);
    v18 = v39 - sub_C444A0(a5);
  }
  if ( v18 <= 1 )
  {
    v19 = 1;
  }
  else
  {
    _BitScanReverse(&v18, v18 - 1);
    v19 = 1 << (32 - (v18 ^ 0x1F));
  }
LABEL_17:
  if ( a4 <= v19 )
    goto LABEL_6;
  v40 = a2;
  v20 = v19;
  while ( 1 )
  {
    switch ( v20 )
    {
      case 1u:
        LOWORD(v21) = 2;
LABEL_31:
        v23 = 0;
        goto LABEL_27;
      case 2u:
        LOWORD(v21) = 3;
        goto LABEL_31;
      case 4u:
        LOWORD(v21) = 4;
        goto LABEL_31;
      case 8u:
        LOWORD(v21) = 5;
        goto LABEL_31;
      case 0x10u:
        LOWORD(v21) = 6;
        goto LABEL_31;
      case 0x20u:
        LOWORD(v21) = 7;
        goto LABEL_31;
      case 0x40u:
        LOWORD(v21) = 8;
        goto LABEL_31;
      case 0x80u:
        LOWORD(v21) = 9;
        goto LABEL_31;
    }
    v21 = sub_3007020(*(_QWORD **)(v43 + 64), v20);
    HIWORD(v7) = HIWORD(v21);
    v23 = v22;
LABEL_27:
    LOWORD(v7) = v21;
    v24 = *(__int64 (**)())(*(_QWORD *)a1 + 1392LL);
    if ( v24 != sub_2FE3480 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v24)(a1, v44, v45, v7, v23) )
      {
        v26 = *(__int64 (**)())(*(_QWORD *)a1 + 1432LL);
        if ( v26 != sub_2FE34A0 )
        {
          v27 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v26)(a1, v7, v23, v44, v45);
          if ( (_BYTE)v27 )
            break;
        }
      }
    }
    v20 = (((((((((v20 | ((unsigned __int64)v20 >> 1)) >> 2) | v20 | ((unsigned __int64)v20 >> 1)) >> 4)
             | ((v20 | ((unsigned __int64)v20 >> 1)) >> 2)
             | v20
             | ((unsigned __int64)v20 >> 1)) >> 8)
           | ((((v20 | ((unsigned __int64)v20 >> 1)) >> 2) | v20 | ((unsigned __int64)v20 >> 1)) >> 4)
           | ((v20 | ((unsigned __int64)v20 >> 1)) >> 2)
           | v20
           | ((unsigned __int64)v20 >> 1)) >> 16)
         | ((((((v20 | ((unsigned __int64)v20 >> 1)) >> 2) | v20 | ((unsigned __int64)v20 >> 1)) >> 4)
           | ((v20 | ((unsigned __int64)v20 >> 1)) >> 2)
           | v20
           | ((unsigned __int64)v20 >> 1)) >> 8)
         | ((((v20 | ((unsigned __int64)v20 >> 1)) >> 2) | v20 | ((unsigned __int64)v20 >> 1)) >> 4)
         | ((v20 | ((unsigned __int64)v20 >> 1)) >> 2)
         | v20
         | (v20 >> 1))
        + 1;
    if ( a4 <= v20 )
      goto LABEL_6;
  }
  v28 = v40;
  v16 = v27;
  v29 = *(_DWORD *)(v40 + 28) & 8;
  if ( v29 )
    v29 = 8;
  v36 = v29;
  *(_QWORD *)&v41 = sub_33FAF80(v43, 216, (__int64)&v46, v7, v23, v29, a7);
  *((_QWORD *)&v41 + 1) = v30;
  *(_QWORD *)&v32 = sub_33FAF80(v43, 216, (__int64)&v46, v7, v23, v31, a7);
  sub_3405C90((_QWORD *)v43, *(_DWORD *)(v28 + 24), (__int64)&v46, v7, v23, v36, a7, v32, v41);
  v34 = sub_33FAF80(v43, 215, (__int64)&v46, v44, v45, v33, a7);
  *(_QWORD *)(a6 + 16) = v28;
  *(_DWORD *)(a6 + 24) = a3;
  *(_QWORD *)(a6 + 32) = v34;
  *(_DWORD *)(a6 + 40) = v35;
LABEL_7:
  if ( v46 )
    sub_B91220((__int64)&v46, v46);
  return v16;
}
