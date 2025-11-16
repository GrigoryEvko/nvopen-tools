// Function: sub_20BD400
// Address: 0x20bd400
//
__int64 *__fastcall sub_20BD400(
        double a1,
        double a2,
        __m128i a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        unsigned __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v12; // rsi
  const void ***v13; // rax
  unsigned int v14; // edx
  __int64 v15; // r14
  unsigned int v16; // ecx
  __int64 v17; // r15
  char v18; // al
  __int64 v19; // rdx
  unsigned int v20; // eax
  unsigned int v21; // ecx
  unsigned int v22; // ebx
  __m128i v23; // xmm0
  int v24; // eax
  unsigned int v25; // ebx
  __int64 v26; // rcx
  char v27; // r15
  unsigned int v28; // eax
  __int64 v29; // rsi
  unsigned int v30; // r14d
  unsigned int v31; // r14d
  unsigned int v32; // eax
  unsigned __int64 v33; // rax
  __int128 v34; // rax
  __int64 *v35; // rax
  unsigned int v36; // edx
  __int128 v37; // rax
  __int64 *v38; // rax
  unsigned int v39; // edx
  unsigned __int8 *v40; // rax
  unsigned int v41; // r14d
  __int128 v42; // rax
  __int64 *v43; // rax
  unsigned int v44; // edx
  __int64 *v45; // r14
  unsigned int v47; // eax
  __int128 v48; // [rsp-10h] [rbp-D0h]
  unsigned int v49; // [rsp+0h] [rbp-C0h]
  __int64 v50; // [rsp+0h] [rbp-C0h]
  unsigned int v51; // [rsp+0h] [rbp-C0h]
  unsigned int v52; // [rsp+8h] [rbp-B8h]
  const void **v53; // [rsp+8h] [rbp-B8h]
  __m128i v56; // [rsp+30h] [rbp-90h] BYREF
  __int64 v57; // [rsp+40h] [rbp-80h] BYREF
  int v58; // [rsp+48h] [rbp-78h]
  char v59[8]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v60; // [rsp+58h] [rbp-68h]
  __m128i v61; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v62; // [rsp+70h] [rbp-50h] BYREF
  const void **v63; // [rsp+78h] [rbp-48h]
  unsigned __int64 v64; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v65; // [rsp+88h] [rbp-38h]
  __int64 v66; // [rsp+D0h] [rbp+10h]
  unsigned __int64 v67; // [rsp+D8h] [rbp+18h]
  unsigned __int64 v68; // [rsp+D8h] [rbp+18h]

  v12 = *(_QWORD *)(a10 + 72);
  v56.m128i_i64[0] = a8;
  v56.m128i_i64[1] = a9;
  v57 = v12;
  if ( v12 )
    sub_1623A60((__int64)&v57, v12, 2);
  v58 = *(_DWORD *)(a10 + 64);
  v13 = (const void ***)(*(_QWORD *)(a6 + 40) + 16LL * (unsigned int)a7);
  v15 = sub_1D323C0(a5, a10, a11, (__int64)&v57, *(unsigned __int8 *)v13, v13[1], a1, a2, *(double *)a3.m128i_i64);
  v16 = v14;
  v66 = v15;
  v17 = v15;
  v52 = v14;
  v67 = v14 | a11 & 0xFFFFFFFF00000000LL;
  if ( v56.m128i_i8[0] )
  {
    switch ( v56.m128i_i8[0] )
    {
      case 0xE:
      case 0xF:
      case 0x10:
      case 0x11:
      case 0x12:
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
      case 0x17:
      case 0x38:
      case 0x39:
      case 0x3A:
      case 0x3B:
      case 0x3C:
      case 0x3D:
        v59[0] = 2;
        v60 = 0;
        break;
      case 0x18:
      case 0x19:
      case 0x1A:
      case 0x1B:
      case 0x1C:
      case 0x1D:
      case 0x1E:
      case 0x1F:
      case 0x20:
      case 0x3E:
      case 0x3F:
      case 0x40:
      case 0x41:
      case 0x42:
      case 0x43:
        v59[0] = 3;
        v60 = 0;
        break;
      case 0x21:
      case 0x22:
      case 0x23:
      case 0x24:
      case 0x25:
      case 0x26:
      case 0x27:
      case 0x28:
      case 0x44:
      case 0x45:
      case 0x46:
      case 0x47:
      case 0x48:
      case 0x49:
        v59[0] = 4;
        v60 = 0;
        break;
      case 0x29:
      case 0x2A:
      case 0x2B:
      case 0x2C:
      case 0x2D:
      case 0x2E:
      case 0x2F:
      case 0x30:
      case 0x4A:
      case 0x4B:
      case 0x4C:
      case 0x4D:
      case 0x4E:
      case 0x4F:
        v59[0] = 5;
        v60 = 0;
        break;
      case 0x31:
      case 0x32:
      case 0x33:
      case 0x34:
      case 0x35:
      case 0x36:
      case 0x50:
      case 0x51:
      case 0x52:
      case 0x53:
      case 0x54:
      case 0x55:
        v59[0] = 6;
        v60 = 0;
        break;
      case 0x37:
        v59[0] = 7;
        v60 = 0;
        break;
      case 0x56:
      case 0x57:
      case 0x58:
      case 0x62:
      case 0x63:
      case 0x64:
        v59[0] = 8;
        v60 = 0;
        break;
      case 0x59:
      case 0x5A:
      case 0x5B:
      case 0x5C:
      case 0x5D:
      case 0x65:
      case 0x66:
      case 0x67:
      case 0x68:
      case 0x69:
        v59[0] = 9;
        v60 = 0;
        break;
      case 0x5E:
      case 0x5F:
      case 0x60:
      case 0x61:
      case 0x6A:
      case 0x6B:
      case 0x6C:
      case 0x6D:
        v59[0] = 10;
        v60 = 0;
        break;
    }
    goto LABEL_28;
  }
  v49 = v14;
  v18 = sub_1F596B0((__int64)&v56);
  v16 = v49;
  v59[0] = v18;
  v60 = v19;
  if ( v18 )
  {
LABEL_28:
    v51 = v16;
    v47 = sub_1F3E310(v59);
    v21 = v51;
    v22 = v47;
    goto LABEL_7;
  }
  v20 = sub_1F58D40((__int64)v59);
  v21 = v49;
  v22 = v20;
LABEL_7:
  v23 = _mm_load_si128(&v56);
  v24 = *(unsigned __int16 *)(v15 + 24);
  v25 = v22 >> 3;
  v61 = v23;
  if ( v24 != 10 && v24 != 32 )
  {
    v26 = *(_QWORD *)(v15 + 40) + 16LL * v21;
    v27 = *(_BYTE *)v26;
    v63 = *(const void ***)(v26 + 8);
    LOBYTE(v62) = v27;
    if ( v61.m128i_i8[0] )
      v28 = word_430A1A0[(unsigned __int8)(v61.m128i_i8[0] - 14)];
    else
      v28 = sub_1F58D30((__int64)&v61);
    v29 = v28 - 1;
    if ( !v28 || (v28 & (unsigned int)v29) != 0 )
    {
      *(_QWORD *)&v37 = sub_1D38BB0((__int64)a5, v29, (__int64)&v57, v62, v63, 0, v23, a2, a3, 0);
      v38 = sub_1D332F0(a5, 116, (__int64)&v57, v62, v63, 0, *(double *)v23.m128i_i64, a2, a3, v15, v67, v37);
      v52 = v39;
      v17 = (__int64)v38;
    }
    else
    {
      _BitScanReverse(&v30, v28);
      v31 = v30 ^ 0x1F;
      if ( v27 )
        v32 = sub_1F3E310(&v62);
      else
        v32 = sub_1F58D40((__int64)&v62);
      v65 = v32;
      if ( v32 > 0x40 )
        sub_16A4EF0((__int64)&v64, 0, 0);
      else
        v64 = 0;
      if ( v31 != 31 )
      {
        v33 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v31 + 33);
        if ( v65 > 0x40 )
          *(_QWORD *)v64 |= v33;
        else
          v64 |= v33;
      }
      *(_QWORD *)&v34 = sub_1D38970((__int64)a5, (__int64)&v64, (__int64)&v57, v62, v63, 0, v23, a2, a3, 0);
      v35 = sub_1D332F0(a5, 118, (__int64)&v57, v62, v63, 0, *(double *)v23.m128i_i64, a2, a3, v66, v67, v34);
      v52 = v36;
      v17 = (__int64)v35;
      if ( v65 > 0x40 && v64 )
        j_j___libc_free_0_0(v64);
    }
  }
  v50 = v52;
  v40 = (unsigned __int8 *)(*(_QWORD *)(v17 + 40) + 16LL * v52);
  v41 = *v40;
  v53 = (const void **)*((_QWORD *)v40 + 1);
  *(_QWORD *)&v42 = sub_1D38BB0((__int64)a5, v25, (__int64)&v57, *v40, v53, 0, v23, a2, a3, 0);
  v68 = v50 | v67 & 0xFFFFFFFF00000000LL;
  v43 = sub_1D332F0(a5, 54, (__int64)&v57, v41, v53, 0, *(double *)v23.m128i_i64, a2, a3, v17, v68, v42);
  *((_QWORD *)&v48 + 1) = v44 | v68 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v48 = v43;
  v45 = sub_1D332F0(a5, 52, (__int64)&v57, v41, v53, 0, *(double *)v23.m128i_i64, a2, a3, a6, a7, v48);
  if ( v57 )
    sub_161E7C0((__int64)&v57, v57);
  return v45;
}
