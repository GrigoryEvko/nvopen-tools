// Function: sub_1D2C2D0
// Address: 0x1d2c2d0
//
__int64 __fastcall sub_1D2C2D0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v11; // rax
  unsigned __int8 *v13; // rax
  int v14; // edx
  __int64 v15; // r8
  unsigned int v16; // ecx
  __int64 v17; // r9
  _QWORD *v18; // rax
  int v19; // edx
  __int64 v20; // r11
  __int64 v21; // r10
  __int64 v22; // r9
  __int64 **v23; // r15
  __int64 *v24; // rsi
  __int64 v25; // rsi
  int v26; // edx
  __int16 v27; // ax
  unsigned __int16 v28; // bx
  int v29; // eax
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r12
  int v36; // r13d
  __int16 v37; // ax
  __int128 v38; // [rsp-20h] [rbp-200h]
  __int128 v39; // [rsp-20h] [rbp-200h]
  __int64 v40; // [rsp+0h] [rbp-1E0h]
  __int64 v41; // [rsp+8h] [rbp-1D8h]
  int v43; // [rsp+10h] [rbp-1D0h]
  _QWORD *v45; // [rsp+18h] [rbp-1C8h]
  int v46; // [rsp+28h] [rbp-1B8h]
  __int64 v47; // [rsp+40h] [rbp-1A0h]
  unsigned __int8 *v50; // [rsp+68h] [rbp-178h] BYREF
  _QWORD v51[7]; // [rsp+70h] [rbp-170h] BYREF
  int v52; // [rsp+A8h] [rbp-138h]
  __int64 *v53[3]; // [rsp+B0h] [rbp-130h] BYREF
  __int16 v54; // [rsp+CAh] [rbp-116h]
  __int64 v55[5]; // [rsp+F8h] [rbp-E8h] BYREF
  _BYTE *v56; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v57; // [rsp+128h] [rbp-B8h]
  _BYTE v58[176]; // [rsp+130h] [rbp-B0h] BYREF

  v11 = *(_QWORD *)(a5 + 40) + 16LL * (unsigned int)a6;
  if ( *(_BYTE *)v11 == (_BYTE)a9 && (*(_QWORD *)(v11 + 8) == a10 || (_BYTE)a9) )
    return sub_1D2BB40(a1, a2, a3, a4, a5, a6, a7, a8, a11);
  v47 = sub_1D29190((__int64)a1, 1u, 0, a10, a5, a6);
  v13 = (unsigned __int8 *)(*(_QWORD *)(a7 + 40) + 16LL * (unsigned int)a8);
  v46 = v14;
  v15 = *((_QWORD *)v13 + 1);
  v16 = *v13;
  v56 = 0;
  LODWORD(v57) = 0;
  v18 = sub_1D2B300(a1, 0x30u, (__int64)&v56, v16, v15, v17);
  v20 = a2;
  v21 = a5;
  v22 = a6;
  if ( v56 )
  {
    v40 = a6;
    v41 = a5;
    v43 = v19;
    v45 = v18;
    sub_161E7C0((__int64)&v56, (__int64)v56);
    v22 = v40;
    v21 = v41;
    v19 = v43;
    v18 = v45;
    v20 = a2;
  }
  v51[6] = v18;
  v51[0] = v20;
  v56 = v58;
  v57 = 0x2000000000LL;
  v51[1] = a3;
  v51[2] = v21;
  v51[3] = v22;
  v51[4] = a7;
  v51[5] = a8;
  v52 = v19;
  sub_16BD430((__int64)&v56, 186);
  sub_16BD4C0((__int64)&v56, v47);
  v23 = (__int64 **)v51;
  do
  {
    v24 = *v23;
    v23 += 2;
    sub_16BD4C0((__int64)&v56, (__int64)v24);
    sub_16BD430((__int64)&v56, *((_DWORD *)v23 - 2));
  }
  while ( v23 != v53 );
  v25 = (unsigned __int8)a9;
  if ( !(_BYTE)a9 )
    v25 = a10;
  sub_16BD4D0((__int64)&v56, v25);
  *((_QWORD *)&v38 + 1) = a10;
  v26 = *(_DWORD *)(a4 + 8);
  *(_QWORD *)&v38 = (unsigned __int8)a9;
  v50 = 0;
  sub_1D189E0((__int64)v53, 186, v26, &v50, v47, v46, v38, a11);
  LOBYTE(v27) = v54 & 0x7F;
  HIBYTE(v27) = ((unsigned __int16)(v54 & 0xF87F) >> 8) | 4;
  v54 = v27;
  v28 = v27 & 0xFF7A;
  if ( v55[0] )
    sub_161E7C0((__int64)v55, v55[0]);
  if ( v50 )
    sub_161E7C0((__int64)&v50, (__int64)v50);
  sub_16BD3E0((__int64)&v56, v28);
  v29 = sub_1E340A0(a11);
  sub_16BD430((__int64)&v56, v29);
  v53[0] = 0;
  v30 = sub_1D17920((__int64)a1, (__int64)&v56, a4, (__int64 *)v53);
  v35 = (__int64)v30;
  if ( v30 )
  {
    sub_1E34340(v30[13], a11, v31, v32, v33, v34);
  }
  else
  {
    v35 = a1[26];
    v36 = *(_DWORD *)(a4 + 8);
    if ( v35 )
      a1[26] = *(_QWORD *)v35;
    else
      v35 = sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v39 + 1) = a10;
    *(_QWORD *)&v39 = (unsigned __int8)a9;
    sub_1D189E0(v35, 186, v36, (unsigned __int8 **)a4, v47, v46, v39, a11);
    v37 = *(_WORD *)(v35 + 26) & 0xF87F;
    HIBYTE(v37) |= 4u;
    *(_WORD *)(v35 + 26) = v37;
    sub_1D23B60((__int64)a1, v35, (__int64)v51, 4);
    sub_16BDA20(a1 + 40, (__int64 *)v35, v53[0]);
    sub_1D172A0((__int64)a1, v35);
  }
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  return v35;
}
