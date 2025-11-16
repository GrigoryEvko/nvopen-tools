// Function: sub_3411C40
// Address: 0x3411c40
//
__int64 __fastcall sub_3411C40(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        __int128 a9)
{
  __int64 v13; // rax
  unsigned __int16 v14; // r8
  unsigned __int16 v15; // dx
  __int64 v16; // rax
  unsigned __int8 *v17; // rax
  __m128i v18; // xmm1
  __int64 v19; // rdx
  __int64 v20; // r9
  unsigned __int8 *v21; // rax
  __int64 v22; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // r9
  char v26; // dl
  char v27; // r10
  __int64 v28; // rax
  __int64 v29; // rdi
  char v30; // dl
  char v31; // al
  unsigned __int64 v32; // rdx
  __m128i v33; // xmm0
  __int128 v34; // [rsp-10h] [rbp-D0h]
  __int128 v35; // [rsp-10h] [rbp-D0h]
  __int64 v36; // [rsp+8h] [rbp-B8h]
  __int64 v37; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v38; // [rsp+10h] [rbp-B0h]
  unsigned __int16 v39; // [rsp+1Ch] [rbp-A4h]
  char v40; // [rsp+1Ch] [rbp-A4h]
  __m128i v41; // [rsp+40h] [rbp-80h] BYREF
  __int16 v42; // [rsp+50h] [rbp-70h]
  __int64 v43; // [rsp+58h] [rbp-68h]
  __int64 v44; // [rsp+60h] [rbp-60h] BYREF
  __int64 v45; // [rsp+68h] [rbp-58h]
  __int64 v46; // [rsp+70h] [rbp-50h]
  __int64 v47; // [rsp+78h] [rbp-48h]
  unsigned __int8 *v48; // [rsp+80h] [rbp-40h]
  __int64 v49; // [rsp+88h] [rbp-38h]

  v13 = *(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4;
  v14 = a9;
  v15 = *(_WORD *)v13;
  v16 = *(_QWORD *)(v13 + 8);
  if ( v15 == (_WORD)a9 )
  {
    if ( (_WORD)a9 || v16 == *((_QWORD *)&a9 + 1) )
      goto LABEL_3;
    v45 = v16;
    LOWORD(v44) = 0;
LABEL_6:
    v37 = a4;
    v39 = a9;
    v24 = sub_3007260((__int64)&v44);
    v14 = v39;
    a4 = v37;
    v25 = v24;
    v27 = v26;
    if ( !v39 )
    {
LABEL_7:
      v36 = a4;
      v38 = v25;
      v40 = v27;
      v28 = sub_3007260((__int64)&a9);
      a4 = v36;
      v25 = v38;
      v29 = v28;
      v31 = v30;
      v27 = v40;
      v32 = v29;
      goto LABEL_8;
    }
    goto LABEL_15;
  }
  LOWORD(v44) = v15;
  v45 = v16;
  if ( !v15 )
    goto LABEL_6;
  if ( v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
LABEL_20:
    BUG();
  v25 = *(_QWORD *)&byte_444C4A0[16 * v15 - 16];
  v27 = byte_444C4A0[16 * v15 - 8];
  if ( !(_WORD)a9 )
    goto LABEL_7;
LABEL_15:
  if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
    goto LABEL_20;
  v32 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
  v31 = byte_444C4A0[16 * v14 - 8];
LABEL_8:
  if ( (v31 || !v27) && v25 < v32 )
  {
    v47 = a4;
    *((_QWORD *)&v35 + 1) = 2;
    v33 = _mm_loadu_si128((const __m128i *)&a9);
    *(_QWORD *)&v35 = &v44;
    v42 = 1;
    v44 = a5;
    v45 = a6;
    v46 = a3;
    v43 = 0;
    v41 = v33;
    v21 = sub_3411BE0(a2, 0x92u, a8, (unsigned __int16 *)&v41, 2, v25, v35);
    goto LABEL_4;
  }
LABEL_3:
  v47 = a4;
  v44 = a5;
  v45 = a6;
  v46 = a3;
  v17 = sub_3400D50((__int64)a2, 0, a8, 1u, a7);
  v18 = _mm_loadu_si128((const __m128i *)&a9);
  v48 = v17;
  v49 = v19;
  *((_QWORD *)&v34 + 1) = 3;
  *(_QWORD *)&v34 = &v44;
  v42 = 1;
  v43 = 0;
  v41 = v18;
  v21 = sub_3411BE0(a2, 0x91u, a8, (unsigned __int16 *)&v41, 2, v20, v34);
LABEL_4:
  *(_QWORD *)a1 = v21;
  *(_QWORD *)(a1 + 16) = v21;
  *(_QWORD *)(a1 + 8) = v22;
  *(_DWORD *)(a1 + 24) = 1;
  return a1;
}
