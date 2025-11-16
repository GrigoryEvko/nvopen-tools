// Function: sub_203DF40
// Address: 0x203df40
//
__int64 *__fastcall sub_203DF40(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned int v5; // r15d
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rcx
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  unsigned __int64 v13; // r13
  char v14; // di
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  char v19; // dl
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned __int8 v22; // al
  __int64 v23; // rdx
  unsigned int v24; // eax
  const void **v25; // r8
  __int64 *v26; // r14
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 v29; // r15
  unsigned __int64 v30; // r13
  __int64 v31; // rax
  __int8 v32; // dl
  __int64 v33; // rax
  unsigned __int8 v34; // al
  __int64 v35; // rax
  const void **v36; // r8
  __int64 v37; // rcx
  __int64 *v38; // rcx
  unsigned int v39; // edx
  unsigned __int64 v40; // rax
  __int64 v41; // rdx
  unsigned __int8 v42; // r15
  __m128i v43; // xmm0
  _QWORD *v44; // r14
  __m128i v45; // xmm1
  __int64 v46; // rcx
  __int64 v47; // r9
  __int64 v48; // rax
  int v49; // edx
  __int64 *v50; // r12
  __int64 v52; // rdx
  const void **v53; // rdx
  const void **v54; // rdx
  __int64 v55; // [rsp+0h] [rbp-140h]
  __int64 v56; // [rsp+8h] [rbp-138h]
  _QWORD *v57; // [rsp+10h] [rbp-130h]
  __int64 v58; // [rsp+18h] [rbp-128h]
  unsigned int v59; // [rsp+18h] [rbp-128h]
  __int64 v60; // [rsp+28h] [rbp-118h]
  __int64 v61; // [rsp+30h] [rbp-110h]
  _QWORD *v62; // [rsp+38h] [rbp-108h]
  unsigned __int64 v63; // [rsp+38h] [rbp-108h]
  int v64; // [rsp+48h] [rbp-F8h]
  __int64 v65; // [rsp+48h] [rbp-F8h]
  unsigned int v66; // [rsp+50h] [rbp-F0h]
  unsigned int v67; // [rsp+58h] [rbp-E8h]
  __int64 v68; // [rsp+58h] [rbp-E8h]
  char v69[8]; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v70; // [rsp+88h] [rbp-B8h]
  char v71[8]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v72; // [rsp+98h] [rbp-A8h]
  __int64 v73; // [rsp+A0h] [rbp-A0h] BYREF
  int v74; // [rsp+A8h] [rbp-98h]
  __m128i v75; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v76; // [rsp+C0h] [rbp-80h]
  __int64 v77; // [rsp+C8h] [rbp-78h]
  __int64 *v78; // [rsp+D0h] [rbp-70h]
  unsigned __int64 v79; // [rsp+D8h] [rbp-68h]
  __m128i v80; // [rsp+E0h] [rbp-60h]
  __int64 *v81; // [rsp+F0h] [rbp-50h]
  unsigned __int64 v82; // [rsp+F8h] [rbp-48h]
  __int64 v83; // [rsp+100h] [rbp-40h]
  int v84; // [rsp+108h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_QWORD *)(v8 + 80);
  v10 = *(_QWORD *)(v8 + 48);
  v11 = *(_QWORD *)(v8 + 40);
  v12 = *(_QWORD *)(v9 + 40) + 16LL * *(unsigned int *)(v8 + 88);
  v13 = *(_QWORD *)(v8 + 88);
  v14 = *(_BYTE *)v12;
  v15 = *(_QWORD *)(v12 + 8);
  v69[0] = v14;
  v16 = *(_QWORD *)(v8 + 200);
  LODWORD(v8) = *(_DWORD *)(v8 + 208);
  v70 = v15;
  v64 = v8;
  v61 = sub_20363F0((__int64)a1, v11, v10);
  v18 = *(_QWORD *)(v61 + 40) + 16LL * (unsigned int)v17;
  v60 = v17;
  v19 = *(_BYTE *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  v71[0] = v19;
  v72 = v20;
  if ( v19 )
    v67 = word_4305480[(unsigned __int8)(v19 - 14)];
  else
    v67 = sub_1F58D30((__int64)v71);
  v21 = *(_QWORD *)(a2 + 72);
  v73 = v21;
  if ( v21 )
    sub_1623A60((__int64)&v73, v21, 2);
  v74 = *(_DWORD *)(a2 + 64);
  if ( v69[0] )
  {
    switch ( v69[0] )
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
        v22 = 2;
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
        v22 = 3;
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
        v22 = 4;
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
        v22 = 5;
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
        v22 = 6;
        break;
      case 0x37:
        v22 = 7;
        break;
      case 0x56:
      case 0x57:
      case 0x58:
      case 0x62:
      case 0x63:
      case 0x64:
        v22 = 8;
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
        v22 = 9;
        break;
      case 0x5E:
      case 0x5F:
      case 0x60:
      case 0x61:
      case 0x6A:
      case 0x6B:
      case 0x6C:
      case 0x6D:
        v22 = 10;
        break;
    }
    v58 = 0;
  }
  else
  {
    v22 = sub_1F596B0((__int64)v69);
    v58 = v23;
  }
  v66 = v22;
  v62 = *(_QWORD **)(a1[1] + 48);
  LOBYTE(v24) = sub_1D15020(v22, v67);
  v25 = 0;
  if ( !(_BYTE)v24 )
  {
    v24 = sub_1F593D0(v62, v66, v58, v67);
    v5 = v24;
    v25 = v54;
  }
  LOBYTE(v5) = v24;
  v26 = sub_2030300(a1, v9, v13, v5, v25, 1, a3, a4, a5);
  v28 = *(_QWORD *)(a2 + 32);
  v63 = v27 | v13 & 0xFFFFFFFF00000000LL;
  v29 = *(_QWORD *)(v28 + 160);
  v30 = *(_QWORD *)(v28 + 168);
  v31 = *(_QWORD *)(v29 + 40) + 16LL * *(unsigned int *)(v28 + 168);
  v32 = *(_BYTE *)v31;
  v33 = *(_QWORD *)(v31 + 8);
  v75.m128i_i8[0] = v32;
  v75.m128i_i64[1] = v33;
  if ( v32 )
  {
    if ( (unsigned __int8)(v32 - 14) > 0x5Fu )
    {
LABEL_13:
      v56 = v75.m128i_i64[1];
      v34 = v75.m128i_i8[0];
      goto LABEL_14;
    }
  }
  else if ( !sub_1F58D20((__int64)&v75) )
  {
    goto LABEL_13;
  }
  v34 = sub_1F7E0F0((__int64)&v75);
  v56 = v52;
LABEL_14:
  v57 = *(_QWORD **)(a1[1] + 48);
  v59 = v34;
  LOBYTE(v35) = sub_1D15020(v34, v67);
  v36 = 0;
  if ( !(_BYTE)v35 )
  {
    v35 = sub_1F593D0(v57, v59, v56, v67);
    v55 = v35;
    v36 = v53;
  }
  v37 = v55;
  LOBYTE(v37) = v35;
  v38 = sub_2030300(a1, v29, v30, v37, v36, 0, a3, a4, a5);
  v40 = v39 | v30 & 0xFFFFFFFF00000000LL;
  v41 = *(_QWORD *)(a2 + 32);
  v42 = *(_BYTE *)(a2 + 88);
  v68 = *(_QWORD *)(a2 + 96);
  v43 = _mm_loadu_si128((const __m128i *)v41);
  v76 = v61;
  v77 = v60;
  v78 = v26;
  v44 = (_QWORD *)a1[1];
  v79 = v63;
  v75 = v43;
  v45 = _mm_loadu_si128((const __m128i *)(v41 + 120));
  v82 = v40;
  v81 = v38;
  v46 = *(_QWORD *)(a2 + 104);
  v83 = v16;
  LODWORD(v40) = v64;
  v80 = v45;
  v65 = v46;
  v84 = v40;
  v48 = sub_1D29190((__int64)v44, 1u, 0, v46, v68, v47);
  v50 = sub_1D24800(v44, v48, v49, v42, v68, (__int64)&v73, v75.m128i_i64, 6, v65);
  if ( v73 )
    sub_161E7C0((__int64)&v73, v73);
  return v50;
}
