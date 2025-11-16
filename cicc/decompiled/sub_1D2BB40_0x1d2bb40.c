// Function: sub_1D2BB40
// Address: 0x1d2bb40
//
__int64 __fastcall sub_1D2BB40(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned __int8 *v11; // rax
  unsigned __int8 *v12; // rax
  int v13; // edx
  __int64 v14; // r8
  unsigned int v15; // ecx
  __int64 v16; // r9
  _QWORD *v17; // rax
  int v18; // edx
  __int64 **v19; // rbx
  __int64 *v20; // rsi
  __int64 v21; // rsi
  int v22; // r11d
  unsigned __int16 v23; // ax
  int v24; // eax
  _QWORD *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r12
  int v32; // r13d
  __int128 v33; // [rsp-20h] [rbp-1F0h]
  __int128 v34; // [rsp-20h] [rbp-1F0h]
  int v35; // [rsp+0h] [rbp-1D0h]
  _QWORD *v36; // [rsp+8h] [rbp-1C8h]
  int v38; // [rsp+20h] [rbp-1B0h]
  unsigned __int16 v40; // [rsp+28h] [rbp-1A8h]
  unsigned __int16 v41; // [rsp+28h] [rbp-1A8h]
  unsigned __int8 v42; // [rsp+3Fh] [rbp-191h]
  __int64 v43; // [rsp+40h] [rbp-190h]
  __int64 v44; // [rsp+48h] [rbp-188h]
  unsigned __int8 *v45; // [rsp+58h] [rbp-178h] BYREF
  _QWORD v46[7]; // [rsp+60h] [rbp-170h] BYREF
  int v47; // [rsp+98h] [rbp-138h]
  __int64 *v48[3]; // [rsp+A0h] [rbp-130h] BYREF
  unsigned __int16 v49; // [rsp+BAh] [rbp-116h]
  __int64 v50[5]; // [rsp+E8h] [rbp-E8h] BYREF
  _BYTE *v51; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v52; // [rsp+118h] [rbp-B8h]
  _BYTE v53[176]; // [rsp+120h] [rbp-B0h] BYREF

  v11 = (unsigned __int8 *)(*(_QWORD *)(a5 + 40) + 16LL * (unsigned int)a6);
  v42 = *v11;
  v44 = *((_QWORD *)v11 + 1);
  v43 = sub_1D29190((__int64)a1, 1u, 0, *v11, a5, a6);
  v12 = (unsigned __int8 *)(*(_QWORD *)(a7 + 40) + 16LL * (unsigned int)a8);
  v38 = v13;
  v14 = *((_QWORD *)v12 + 1);
  v15 = *v12;
  v51 = 0;
  LODWORD(v52) = 0;
  v17 = sub_1D2B300(a1, 0x30u, (__int64)&v51, v15, v14, v16);
  if ( v51 )
  {
    v35 = v18;
    v36 = v17;
    sub_161E7C0((__int64)&v51, (__int64)v51);
    v18 = v35;
    v17 = v36;
  }
  v46[6] = v17;
  v46[2] = a5;
  v46[0] = a2;
  v46[1] = a3;
  v51 = v53;
  v46[3] = a6;
  v52 = 0x2000000000LL;
  v46[4] = a7;
  v46[5] = a8;
  v47 = v18;
  sub_16BD430((__int64)&v51, 186);
  sub_16BD4C0((__int64)&v51, v43);
  v19 = (__int64 **)v46;
  do
  {
    v20 = *v19;
    v19 += 2;
    sub_16BD4C0((__int64)&v51, (__int64)v20);
    sub_16BD430((__int64)&v51, *((_DWORD *)v19 - 2));
  }
  while ( v19 != v48 );
  v21 = v42;
  if ( !v42 )
    v21 = v44;
  sub_16BD4D0((__int64)&v51, v21);
  v22 = *(_DWORD *)(a4 + 8);
  v45 = 0;
  *((_QWORD *)&v33 + 1) = v44;
  *(_QWORD *)&v33 = v42;
  sub_1D189E0((__int64)v48, 186, v22, &v45, v43, v38, v33, a9);
  v23 = v49 & 0xF87F;
  v49 = v23;
  LOBYTE(v23) = v23 & 0x7A;
  if ( v50[0] )
  {
    v40 = v23;
    sub_161E7C0((__int64)v50, v50[0]);
    v23 = v40;
  }
  if ( v45 )
  {
    v41 = v23;
    sub_161E7C0((__int64)&v45, (__int64)v45);
    v23 = v41;
  }
  sub_16BD3E0((__int64)&v51, v23);
  v24 = sub_1E340A0(a9);
  sub_16BD430((__int64)&v51, v24);
  v48[0] = 0;
  v25 = sub_1D17920((__int64)a1, (__int64)&v51, a4, (__int64 *)v48);
  v30 = (__int64)v25;
  if ( v25 )
  {
    sub_1E34340(v25[13], a9, v26, v27, v28, v29);
  }
  else
  {
    v30 = a1[26];
    v32 = *(_DWORD *)(a4 + 8);
    if ( v30 )
      a1[26] = *(_QWORD *)v30;
    else
      v30 = sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v34 + 1) = v44;
    *(_QWORD *)&v34 = v42;
    sub_1D189E0(v30, 186, v32, (unsigned __int8 **)a4, v43, v38, v34, a9);
    *(_WORD *)(v30 + 26) &= 0xF87Fu;
    sub_1D23B60((__int64)a1, v30, (__int64)v46, 4);
    sub_16BDA20(a1 + 40, (__int64 *)v30, v48[0]);
    sub_1D172A0((__int64)a1, v30);
  }
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  return v30;
}
