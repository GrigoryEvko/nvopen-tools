// Function: sub_2074D20
// Address: 0x2074d20
//
__int64 *__fastcall sub_2074D20(__int64 a1, __int64 a2, char a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 *v13; // r14
  __int64 v14; // rdx
  __int64 *v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // r14d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // r15
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rsi
  int v26; // r9d
  __int64 v27; // rdi
  _QWORD *v28; // r13
  __int64 *v29; // r15
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // r15
  __int64 v37; // r14
  int v38; // edx
  int v39; // r9d
  __int64 v40; // r10
  __int32 v41; // eax
  unsigned int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rax
  int v45; // r8d
  int v46; // r9d
  __int64 v47; // r15
  int v48; // edx
  int v49; // r14d
  __int64 *result; // rax
  __int64 v51; // rsi
  int v52; // eax
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 v55; // rax
  __m128i v56; // xmm0
  char v57; // al
  __int64 v58; // rax
  __int64 v59; // rax
  _QWORD *v60; // rax
  int v61; // eax
  int v62; // eax
  int v63; // [rsp+8h] [rbp-128h]
  int v64; // [rsp+10h] [rbp-120h]
  __int64 v65; // [rsp+10h] [rbp-120h]
  __int64 v66; // [rsp+18h] [rbp-118h]
  unsigned __int64 v67; // [rsp+20h] [rbp-110h]
  int v68; // [rsp+28h] [rbp-108h]
  char v69; // [rsp+28h] [rbp-108h]
  int v70; // [rsp+28h] [rbp-108h]
  __int64 v71; // [rsp+28h] [rbp-108h]
  __int128 v72; // [rsp+30h] [rbp-100h]
  __int128 v73; // [rsp+40h] [rbp-F0h]
  int v74; // [rsp+54h] [rbp-DCh]
  __int128 v75; // [rsp+60h] [rbp-D0h]
  __int64 *v77; // [rsp+78h] [rbp-B8h]
  __int64 v78; // [rsp+90h] [rbp-A0h] BYREF
  int v79; // [rsp+98h] [rbp-98h]
  __int64 v80; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v81; // [rsp+A8h] [rbp-88h]
  __m128i v82; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v83; // [rsp+C0h] [rbp-70h]
  __int128 v84; // [rsp+D0h] [rbp-60h] BYREF
  __m128i v85; // [rsp+E0h] [rbp-50h]
  __int64 v86; // [rsp+F0h] [rbp-40h]

  v8 = *(_QWORD *)a1;
  v9 = *(_DWORD *)(a1 + 536);
  v78 = 0;
  v79 = v9;
  if ( v8 )
  {
    if ( &v78 != (__int64 *)(v8 + 48) )
    {
      v10 = *(_QWORD *)(v8 + 48);
      v78 = v10;
      if ( v10 )
        sub_1623A60((__int64)&v78, v10, 2);
    }
  }
  v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v77 = *(__int64 **)(a2 - 24 * v11);
  v12 = *(_QWORD *)(a2 + 24 * (1 - v11));
  v13 = *(__int64 **)(a2 + 24 * (2 - v11));
  if ( a3 )
  {
    *(_QWORD *)&v73 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), a4, a5, a6);
    *((_QWORD *)&v73 + 1) = v14;
    v15 = sub_20685E0(a1, v13, a4, a5, a6);
    v17 = v16;
    *(_QWORD *)&v75 = v15;
    *((_QWORD *)&v75 + 1) = v16;
    *(_QWORD *)&v72 = sub_20685E0(a1, (__int64 *)v12, a4, a5, a6);
    v18 = v15[5] + 16LL * v17;
    *((_QWORD *)&v72 + 1) = v19;
    LOBYTE(v19) = *(_BYTE *)v18;
    v20 = *(_QWORD *)(v18 + 8);
    LOBYTE(v80) = v19;
    v81 = v20;
LABEL_7:
    v74 = sub_1D172F0(*(_QWORD *)(a1 + 552), (unsigned int)v80, v81);
    goto LABEL_8;
  }
  v28 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v28 = (_QWORD *)*v28;
  v74 = (int)v28;
  v29 = *(__int64 **)(a2 + 24 * (3 - v11));
  *(_QWORD *)&v73 = sub_20685E0(a1, v77, a4, a5, a6);
  *((_QWORD *)&v73 + 1) = v30;
  *(_QWORD *)&v75 = sub_20685E0(a1, v29, a4, a5, a6);
  *((_QWORD *)&v75 + 1) = v31;
  *(_QWORD *)&v72 = sub_20685E0(a1, v13, a4, a5, a6);
  *((_QWORD *)&v72 + 1) = v32;
  v33 = *(_QWORD *)(v75 + 40) + 16LL * DWORD2(v75);
  LOBYTE(v32) = *(_BYTE *)v33;
  v34 = *(_QWORD *)(v33 + 8);
  LOBYTE(v80) = v32;
  v81 = v34;
  if ( !(_DWORD)v28 )
    goto LABEL_7;
LABEL_8:
  v82 = 0u;
  v83 = 0;
  sub_14A8180(a2, v82.m128i_i64, 0);
  v21 = *(_QWORD *)(a2 + 48);
  if ( v21 || *(__int16 *)(a2 + 18) < 0 )
    LODWORD(v21) = sub_1625790(a2, 4);
  v22 = *(_QWORD *)(a1 + 568);
  if ( !v22 )
    goto LABEL_17;
  v68 = v21;
  v23 = 1;
  v24 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  v25 = *(_QWORD *)a2;
  v26 = v68;
  v27 = v24;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v25 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v58 = *(_QWORD *)(v25 + 32);
        v25 = *(_QWORD *)(v25 + 24);
        v23 *= v58;
        continue;
      case 1:
        v55 = 16;
        break;
      case 2:
        v55 = 32;
        break;
      case 3:
      case 9:
        v55 = 64;
        break;
      case 4:
        v55 = 80;
        break;
      case 5:
      case 6:
        v55 = 128;
        break;
      case 7:
        v62 = sub_15A9520(v27, 0);
        v26 = v68;
        v55 = (unsigned int)(8 * v62);
        break;
      case 0xB:
        v55 = *(_DWORD *)(v25 + 8) >> 8;
        break;
      case 0xD:
        v60 = (_QWORD *)sub_15A9930(v27, v25);
        v26 = v68;
        v55 = 8LL * *v60;
        break;
      case 0xE:
        v63 = v68;
        v65 = *(_QWORD *)(v25 + 24);
        v71 = *(_QWORD *)(v25 + 32);
        v67 = (unsigned int)sub_15A9FE0(v27, v65);
        v59 = sub_127FA20(v27, v65);
        v26 = v63;
        v55 = 8 * v71 * v67 * ((v67 + ((unsigned __int64)(v59 + 7) >> 3) - 1) / v67);
        break;
      case 0xF:
        v61 = sub_15A9520(v27, *(_DWORD *)(v25 + 8) >> 8);
        v26 = v68;
        v55 = (unsigned int)(8 * v61);
        break;
    }
    break;
  }
  v56 = _mm_load_si128(&v82);
  v70 = v26;
  v86 = v83;
  *(_QWORD *)&v84 = v77;
  *((_QWORD *)&v84 + 1) = (unsigned __int64)(v55 * v23 + 7) >> 3;
  v85 = v56;
  v57 = sub_134CBB0(v22, (__int64)&v84, 0);
  LODWORD(v21) = v70;
  if ( v57 )
  {
    v35 = *(_QWORD **)(a1 + 552);
    v69 = 0;
    v37 = 0;
    v36 = (__int64)(v35 + 11);
  }
  else
  {
LABEL_17:
    v35 = *(_QWORD **)(a1 + 552);
    v69 = 1;
    v36 = v35[22];
    v37 = v35[23];
  }
  if ( (_BYTE)v80 )
  {
    v38 = sub_2045180(v80);
  }
  else
  {
    v64 = v21;
    v66 = v35[4];
    v52 = sub_1F58D40((__int64)&v80);
    v39 = v64;
    v40 = v66;
    v38 = v52;
  }
  v85.m128i_i8[0] = 0;
  v41 = 0;
  v42 = (unsigned int)(v38 + 7) >> 3;
  v84 = (unsigned __int64)v77;
  if ( v77 )
  {
    v43 = *v77;
    if ( *(_BYTE *)(*v77 + 8) == 16 )
      v43 = **(_QWORD **)(v43 + 16);
    v41 = *(_DWORD *)(v43 + 8) >> 8;
  }
  v85.m128i_i32[1] = v41;
  v44 = sub_1E0B8E0(v40, 1u, v42, v74, (int)&v82, v39, v84, v85.m128i_i64[0], 1u, 0, 0);
  v47 = sub_1D257D0(
          *(_QWORD **)(a1 + 552),
          (unsigned int)v80,
          v81,
          (__int64)&v78,
          v36,
          v37,
          v73,
          v72,
          v75,
          v80,
          v81,
          v44,
          0,
          a3);
  v49 = v48;
  if ( v69 )
  {
    v53 = *(unsigned int *)(a1 + 112);
    if ( (unsigned int)v53 >= *(_DWORD *)(a1 + 116) )
    {
      sub_16CD150(a1 + 104, (const void *)(a1 + 120), 0, 16, v45, v46);
      v53 = *(unsigned int *)(a1 + 112);
    }
    v54 = (__int64 *)(*(_QWORD *)(a1 + 104) + 16 * v53);
    *v54 = v47;
    v54[1] = 1;
    ++*(_DWORD *)(a1 + 112);
  }
  *(_QWORD *)&v84 = a2;
  result = sub_205F5C0(a1 + 8, (__int64 *)&v84);
  v51 = v78;
  result[1] = v47;
  *((_DWORD *)result + 4) = v49;
  if ( v51 )
    return (__int64 *)sub_161E7C0((__int64)&v78, v51);
  return result;
}
