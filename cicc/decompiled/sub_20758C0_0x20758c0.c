// Function: sub_20758C0
// Address: 0x20758c0
//
void __fastcall sub_20758C0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  _BYTE *v14; // r15
  __int64 v15; // rax
  int v16; // eax
  bool v17; // zf
  unsigned int v18; // ecx
  __int64 v19; // rdx
  int v20; // eax
  unsigned int v21; // ecx
  unsigned int v22; // edx
  __int64 v23; // rdi
  __int64 v24; // r10
  int v25; // eax
  __int64 v26; // r10
  int v27; // ecx
  int v28; // edx
  __int64 *v29; // rax
  unsigned int v30; // edx
  __int64 v31; // rax
  _QWORD *v32; // r11
  __int64 v33; // r9
  __int64 (__fastcall *v34)(__int64, __int64); // rax
  __int64 v35; // rsi
  unsigned int v36; // ecx
  unsigned __int64 v37; // r13
  __int128 v38; // rax
  __int64 *v39; // r12
  int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // r13
  unsigned int v43; // eax
  __int64 v44; // rax
  unsigned int v45; // edx
  __int128 v46; // [rsp-20h] [rbp-110h]
  __int64 v47; // [rsp+0h] [rbp-F0h]
  int v48; // [rsp+0h] [rbp-F0h]
  unsigned __int8 v49; // [rsp+8h] [rbp-E8h]
  int v50; // [rsp+8h] [rbp-E8h]
  __int64 v51; // [rsp+8h] [rbp-E8h]
  int v52; // [rsp+10h] [rbp-E0h]
  _QWORD *v53; // [rsp+10h] [rbp-E0h]
  __int64 v54; // [rsp+10h] [rbp-E0h]
  __int64 v55; // [rsp+18h] [rbp-D8h]
  unsigned __int8 v56; // [rsp+18h] [rbp-D8h]
  __int64 v57; // [rsp+18h] [rbp-D8h]
  __int64 v58; // [rsp+60h] [rbp-90h] BYREF
  int v59; // [rsp+68h] [rbp-88h]
  __int64 v60; // [rsp+70h] [rbp-80h] BYREF
  __int64 v61; // [rsp+78h] [rbp-78h]
  __int128 v62; // [rsp+80h] [rbp-70h]
  __int64 v63; // [rsp+90h] [rbp-60h]
  __int64 v64[10]; // [rsp+A0h] [rbp-50h] BYREF

  v7 = *(_DWORD *)(a1 + 536);
  v8 = *(_QWORD *)a1;
  v58 = 0;
  v59 = v7;
  if ( v8 )
  {
    if ( &v58 != (__int64 *)(v8 + 48) )
    {
      v9 = *(_QWORD *)(v8 + 48);
      v58 = v9;
      if ( v9 )
        sub_1623A60((__int64)&v58, v9, 2);
    }
  }
  v52 = (*(unsigned __int16 *)(a2 + 18) >> 7) & 7;
  v49 = *(_BYTE *)(a2 + 56);
  v10 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v11 = *(_QWORD *)(a1 + 552);
  v13 = v12;
  v14 = *(_BYTE **)(v11 + 16);
  v55 = *(_QWORD *)a2;
  v15 = sub_1E0A0C0(*(_QWORD *)(v11 + 32));
  LOBYTE(v16) = sub_204D4D0((__int64)v14, v15, v55);
  v17 = v14[108] == 0;
  v18 = *(unsigned __int16 *)(a2 + 18);
  LODWORD(v60) = v16;
  v61 = v19;
  if ( v17 )
  {
    if ( (_BYTE)v60 )
    {
      v20 = sub_2045180(v60);
      v22 = v21;
    }
    else
    {
      v48 = 1 << (v18 >> 1) >> 1;
      v20 = sub_1F58D40((__int64)&v60);
      v22 = v48;
      v21 = v48;
    }
    if ( (unsigned int)(v20 + 7) >> 3 > v22 )
      sub_16BD130("Cannot generate unaligned atomic load", 1u);
  }
  else
  {
    v22 = 1 << (v18 >> 1) >> 1;
    v21 = v22;
  }
  v23 = *(_QWORD *)(a1 + 552);
  v24 = *(_QWORD *)(v23 + 32);
  v56 = v49;
  memset(v64, 0, 24);
  if ( !v22 )
  {
    v51 = v24;
    v43 = sub_1D172F0(v23, (unsigned int)v60, v61);
    v24 = v51;
    v21 = v43;
  }
  if ( (_BYTE)v60 )
  {
    v28 = sub_2045180(v60);
  }
  else
  {
    v47 = v24;
    v50 = v21;
    v25 = sub_1F58D40((__int64)&v60);
    v26 = v47;
    v27 = v50;
    v28 = v25;
  }
  v29 = *(__int64 **)(a2 - 24);
  v30 = (unsigned int)(v28 + 7) >> 3;
  if ( v29 )
  {
    v62 = *(unsigned __int64 *)(a2 - 24);
    v29 = (__int64 *)*v29;
    v17 = *((_BYTE *)v29 + 8) == 16;
    LOBYTE(v63) = 0;
    if ( v17 )
      v29 = *(__int64 **)v29[2];
    LODWORD(v29) = *((_DWORD *)v29 + 2) >> 8;
  }
  else
  {
    v62 = 0u;
    v63 = 0;
  }
  HIDWORD(v63) = (_DWORD)v29;
  v31 = sub_1E0B8E0(v26, 5u, v30, v27, (int)v64, 0, v62, v63, v56, v52, 0);
  v32 = *(_QWORD **)(a1 + 552);
  v33 = v31;
  v34 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 1288LL);
  if ( v34 == sub_2043C60 )
  {
    v35 = (__int64)v10;
    v36 = v13;
  }
  else
  {
    v54 = v33;
    v44 = ((__int64 (__fastcall *)(_BYTE *, __int64 *, __int64, __int64 *, _QWORD))v34)(
            v14,
            v10,
            v13,
            &v58,
            *(_QWORD *)(a1 + 552));
    v32 = *(_QWORD **)(a1 + 552);
    v33 = v54;
    v36 = v45;
    v35 = v44;
  }
  v57 = v33;
  v37 = v36 | v13 & 0xFFFFFFFF00000000LL;
  v53 = v32;
  *(_QWORD *)&v38 = sub_20685E0(a1, *(__int64 **)(a2 - 24), a3, a4, a5);
  *((_QWORD *)&v46 + 1) = v37;
  *(_QWORD *)&v46 = v35;
  v39 = sub_1D25480(v53, 0xDBu, (__int64)&v58, v60, v61, v57, v60, v61, v46, v38);
  LODWORD(v37) = v40;
  v64[0] = a2;
  v41 = sub_205F5C0(a1 + 8, v64);
  v41[1] = (__int64)v39;
  *((_DWORD *)v41 + 4) = v37;
  v42 = *(_QWORD *)(a1 + 552);
  if ( v39 )
  {
    nullsub_686();
    *(_QWORD *)(v42 + 176) = v39;
    *(_DWORD *)(v42 + 184) = 1;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v42 + 176) = 0;
    *(_DWORD *)(v42 + 184) = 1;
  }
  if ( v58 )
    sub_161E7C0((__int64)&v58, v58);
}
