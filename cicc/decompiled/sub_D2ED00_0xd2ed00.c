// Function: sub_D2ED00
// Address: 0xd2ed00
//
__int64 __fastcall sub_D2ED00(
        __int64 a1,
        _QWORD *a2,
        __int64 (__fastcall *a3)(__int64, __int64),
        __int64 a4,
        __int64 a5)
{
  _QWORD *v5; // r12
  unsigned __int64 v6; // rax
  __int64 v7; // r15
  __int64 *v8; // r13
  char *v9; // rax
  size_t v10; // rdx
  _QWORD *v11; // r12
  _QWORD *i; // r14
  _BYTE *v13; // rsi
  unsigned __int64 v14; // rax
  _QWORD *v15; // r15
  _QWORD *v16; // r12
  __int64 v17; // r13
  __int64 v18; // r9
  char v19; // dl
  __int64 v20; // rax
  __int64 v21; // r13
  unsigned __int64 v22; // rdx
  __int64 result; // rax
  __int64 v25; // [rsp+8h] [rbp-198h]
  __int64 v28; // [rsp+38h] [rbp-168h] BYREF
  _BYTE *v29; // [rsp+40h] [rbp-160h] BYREF
  __int64 v30; // [rsp+48h] [rbp-158h]
  _BYTE v31[128]; // [rsp+50h] [rbp-150h] BYREF
  __int64 v32; // [rsp+D0h] [rbp-D0h] BYREF
  char *v33; // [rsp+D8h] [rbp-C8h]
  __int64 v34; // [rsp+E0h] [rbp-C0h]
  int v35; // [rsp+E8h] [rbp-B8h]
  char j; // [rsp+ECh] [rbp-B4h]
  char v37; // [rsp+F0h] [rbp-B0h] BYREF

  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 128) = a1 + 144;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 224) = a1 + 240;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  *(_QWORD *)(a1 + 136) = 0x400000000LL;
  v25 = a1 + 176;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 232) = 0x400000000LL;
  *(_QWORD *)(a1 + 272) = a1 + 288;
  *(_QWORD *)(a1 + 352) = a1 + 368;
  *(_QWORD *)(a1 + 400) = a1 + 416;
  *(_QWORD *)(a1 + 432) = a1 + 448;
  *(_QWORD *)(a1 + 360) = 0x400000000LL;
  *(_QWORD *)(a1 + 440) = 0x1000000000LL;
  *(_QWORD *)(a1 + 648) = 0x400000000LL;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 640) = a1 + 656;
  v5 = (_QWORD *)a2[4];
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  for ( *(_DWORD *)(a1 + 632) = 0; a2 + 3 != v5; v5 = (_QWORD *)v5[1] )
  {
    v7 = 0;
    if ( v5 )
      v7 = (__int64)(v5 - 7);
    if ( !sub_B2FC80(v7) )
    {
      v8 = (__int64 *)a3(a4, v7);
      if ( sub_981210(*v8, v7, (unsigned int *)&v32) || (v9 = (char *)sub_BD5D20(v7), sub_97F890(*v8, v9, v10)) )
      {
        v32 = v7;
        sub_D2EA30(a1 + 608, &v32);
      }
      if ( (*(_BYTE *)(v7 + 32) & 0xFu) - 7 > 1 )
      {
        v6 = sub_D29010(a1, v7);
        sub_D25660(a1 + 128, v25, v6, 0);
      }
    }
  }
  v11 = (_QWORD *)a2[6];
  for ( i = a2 + 5; i != v11; v11 = (_QWORD *)v11[1] )
  {
    while ( 1 )
    {
      if ( !v11 )
        BUG();
      if ( (*(_BYTE *)(v11 - 2) & 0xFu) - 7 > 1 )
      {
        v13 = (_BYTE *)*(v11 - 10);
        if ( !*v13 )
          break;
      }
      v11 = (_QWORD *)v11[1];
      if ( i == v11 )
        goto LABEL_20;
    }
    v14 = sub_D29010(a1, (__int64)v13);
    sub_D25660(a1 + 128, v25, v14, 0);
  }
LABEL_20:
  v32 = 0;
  v29 = v31;
  v30 = 0x1000000000LL;
  v33 = &v37;
  v34 = 16;
  v15 = (_QWORD *)a2[2];
  v16 = a2 + 1;
  v35 = 0;
  for ( j = 1; v16 != v15; v15 = (_QWORD *)v15[1] )
  {
    while ( 1 )
    {
      v17 = (__int64)(v15 - 7);
      if ( !v15 )
        v17 = 0;
      if ( !sub_B2FC80(v17) )
      {
        sub_AE6EC0((__int64)&v32, *(_QWORD *)(v17 - 32));
        if ( v19 )
          break;
      }
      v15 = (_QWORD *)v15[1];
      if ( v16 == v15 )
        goto LABEL_30;
    }
    v20 = (unsigned int)v30;
    v21 = *(_QWORD *)(v17 - 32);
    v22 = (unsigned int)v30 + 1LL;
    if ( v22 > HIDWORD(v30) )
    {
      sub_C8D5F0((__int64)&v29, v31, v22, 8u, a5, v18);
      v20 = (unsigned int)v30;
    }
    *(_QWORD *)&v29[8 * v20] = v21;
    LODWORD(v30) = v30 + 1;
  }
LABEL_30:
  v28 = a1;
  result = sub_D24710(
             (__int64)&v29,
             (__int64)&v32,
             (void (__fastcall *)(__int64, __int64))sub_D29730,
             (__int64)&v28,
             a5);
  if ( !j )
    result = _libc_free(v33, &v32);
  if ( v29 != v31 )
    return _libc_free(v29, &v32);
  return result;
}
