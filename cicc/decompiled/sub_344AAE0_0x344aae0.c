// Function: sub_344AAE0
// Address: 0x344aae0
//
__int64 __fastcall sub_344AAE0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        int a8,
        __int64 a9,
        __int64 a10)
{
  int v12; // eax
  __int64 result; // rax
  const __m128i *v14; // rax
  __int64 v15; // r12
  int v16; // ecx
  __int64 v17; // rsi
  unsigned __int16 *v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // rsi
  int v21; // eax
  unsigned int v22; // esi
  unsigned __int64 *v23; // r8
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  int v27; // eax
  unsigned int v28; // r10d
  int v29; // ecx
  unsigned __int64 v30; // rax
  unsigned int v31; // r9d
  unsigned __int64 v32; // rax
  int v33; // eax
  int v34; // eax
  int v35; // eax
  __int64 (*v36)(); // rax
  __int64 v37; // r12
  unsigned int v38; // eax
  unsigned __int64 v39; // rdx
  __int128 v40; // rax
  __int64 v41; // r9
  __int64 v42; // rdx
  __int128 v43; // rax
  __int64 v44; // r9
  int v45; // eax
  int v46; // eax
  int v47; // eax
  unsigned int v48; // [rsp+8h] [rbp-88h]
  __int128 v49; // [rsp+10h] [rbp-80h]
  unsigned int v50; // [rsp+20h] [rbp-70h]
  unsigned int v51; // [rsp+20h] [rbp-70h]
  unsigned int v52; // [rsp+20h] [rbp-70h]
  unsigned int v53; // [rsp+20h] [rbp-70h]
  unsigned int v54; // [rsp+20h] [rbp-70h]
  unsigned int v55; // [rsp+24h] [rbp-6Ch]
  __int64 v56; // [rsp+28h] [rbp-68h]
  __int64 v57; // [rsp+30h] [rbp-60h]
  __int64 v58; // [rsp+30h] [rbp-60h]
  int v59; // [rsp+30h] [rbp-60h]
  unsigned int v60; // [rsp+30h] [rbp-60h]
  int v61; // [rsp+30h] [rbp-60h]
  unsigned int v62; // [rsp+30h] [rbp-60h]
  __int128 v63; // [rsp+30h] [rbp-60h]
  unsigned __int64 v64; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v65; // [rsp+48h] [rbp-48h]
  unsigned __int64 v66; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v67; // [rsp+58h] [rbp-38h]

  v12 = *(_DWORD *)(a7 + 24);
  if ( v12 != 35 && v12 != 11 )
    return 0;
  if ( *(_DWORD *)(a4 + 24) != 56 )
    return 0;
  v14 = *(const __m128i **)(a4 + 40);
  v15 = v14[2].m128i_i64[1];
  v16 = *(_DWORD *)(v15 + 24);
  if ( v16 != 35 && v16 != 11 )
    return 0;
  v17 = *(_QWORD *)(a7 + 96);
  v49 = (__int128)_mm_loadu_si128(v14);
  v18 = (unsigned __int16 *)(*(_QWORD *)(v14->m128i_i64[0] + 48) + 16LL * v14->m128i_u32[2]);
  v56 = *((_QWORD *)v18 + 1);
  v19 = *v18;
  v65 = *(_DWORD *)(v17 + 32);
  if ( v65 > 0x40 )
  {
    v59 = a6;
    sub_C43780((__int64)&v64, (const void **)(v17 + 24));
    a6 = v59;
  }
  else
  {
    v64 = *(_QWORD *)(v17 + 24);
  }
  switch ( a6 )
  {
    case 12:
      v55 = 17;
      goto LABEL_19;
    case 13:
      sub_C46A40((__int64)&v64, 1);
      v55 = 17;
      goto LABEL_19;
    case 10:
      sub_C46A40((__int64)&v64, 1);
      v55 = 22;
LABEL_19:
      v20 = *(_QWORD *)(v15 + 96);
      v67 = *(_DWORD *)(v20 + 32);
      if ( v67 > 0x40 )
        sub_C43780((__int64)&v66, (const void **)(v20 + 24));
      else
        v66 = *(_QWORD *)(v20 + 24);
      v21 = sub_C49970((__int64)&v64, &v66);
      v22 = v65;
      v23 = &v64;
      if ( v21 > 0 )
      {
        if ( v65 > 0x40 )
        {
          v51 = v65;
          v35 = sub_C44630((__int64)&v64);
          v23 = &v64;
          v22 = v51;
          if ( v35 != 1 )
            goto LABEL_60;
        }
        else
        {
          v24 = v64;
          if ( !v64 || (v64 & (v64 - 1)) != 0 )
            goto LABEL_25;
        }
        v28 = v67;
        if ( v67 > 0x40 )
        {
          v52 = v67;
          v45 = sub_C44630((__int64)&v66);
          v28 = v52;
          v23 = &v64;
          if ( v45 == 1 )
            goto LABEL_39;
        }
        else if ( v66 && (v66 & (v66 - 1)) == 0 )
        {
          goto LABEL_39;
        }
      }
      if ( v22 <= 0x40 )
      {
        v24 = v64;
LABEL_25:
        v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & ~v24;
        if ( !v22 )
          v25 = 0;
        v64 = v25;
        goto LABEL_28;
      }
LABEL_60:
      sub_C43D10((__int64)&v64);
LABEL_28:
      sub_C46250((__int64)&v64);
      if ( v67 > 0x40 )
      {
        sub_C43D10((__int64)&v66);
      }
      else
      {
        v26 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v67) & ~v66;
        if ( !v67 )
          v26 = 0;
        v66 = v26;
      }
      sub_C46250((__int64)&v66);
      v55 = sub_33CBD40(v55, v19, v56);
      v27 = sub_C49970((__int64)&v64, &v66);
      v23 = &v64;
      if ( v27 <= 0 )
      {
LABEL_67:
        v28 = v67;
LABEL_46:
        result = 0;
LABEL_47:
        if ( v28 > 0x40 && v66 )
        {
          v58 = result;
          j_j___libc_free_0_0(v66);
          result = v58;
        }
        goto LABEL_15;
      }
      v22 = v65;
      v28 = v67;
      if ( v65 > 0x40 )
      {
        v48 = v67;
        v53 = v65;
        v46 = sub_C44630((__int64)&v64);
        v23 = &v64;
        v22 = v53;
        v28 = v48;
        if ( v46 != 1 )
          goto LABEL_46;
      }
      else if ( !v64 || (v64 & (v64 - 1)) != 0 )
      {
        goto LABEL_46;
      }
      if ( v28 > 0x40 )
      {
        v54 = v28;
        v47 = sub_C44630((__int64)&v66);
        v28 = v54;
        v23 = &v64;
        if ( v47 != 1 )
          goto LABEL_46;
      }
      else if ( !v66 || (v66 & (v66 - 1)) != 0 )
      {
        goto LABEL_46;
      }
LABEL_39:
      if ( v22 > 0x40 )
      {
        v60 = v28;
        v33 = sub_C444A0((__int64)&v64);
        v28 = v60;
        v29 = v22 - 1 - v33;
      }
      else
      {
        v29 = -1;
        if ( v64 )
        {
          _BitScanReverse64(&v30, v64);
          v29 = 63 - (v30 ^ 0x3F);
        }
      }
      if ( v28 > 0x40 )
      {
        v50 = v28;
        v61 = v29;
        v34 = sub_C444A0((__int64)&v66);
        v28 = v50;
        v29 = v61;
        v31 = v50 - v34;
      }
      else
      {
        v31 = 0;
        if ( v66 )
        {
          _BitScanReverse64(&v32, v66);
          v31 = 64 - (v32 ^ 0x3F);
        }
      }
      if ( v31 != v29 )
        goto LABEL_46;
      v36 = *(__int64 (**)())(*(_QWORD *)a1 + 424LL);
      if ( v36 == sub_2FE3060 )
        goto LABEL_46;
      v62 = v31;
      v37 = *(_QWORD *)(a9 + 16);
      if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, unsigned __int64 *))v36)(
             a1,
             v19,
             v56,
             v31,
             v23) )
      {
        v38 = sub_327FC40(*(_QWORD **)(v37 + 64), v62);
        *(_QWORD *)&v40 = sub_33F7D60((_QWORD *)v37, v38, v39);
        *(_QWORD *)&v63 = sub_3406EB0((_QWORD *)v37, 0xDEu, a10, v19, v56, v41, v49, v40);
        *((_QWORD *)&v63 + 1) = v42;
        *(_QWORD *)&v43 = sub_33ED040((_QWORD *)v37, v55);
        result = sub_340F900((_QWORD *)v37, 0xD0u, a10, a2, a3, v44, v63, v49, v43);
        v28 = v67;
        goto LABEL_47;
      }
      goto LABEL_67;
    case 11:
      v55 = 22;
      goto LABEL_19;
  }
  result = 0;
LABEL_15:
  if ( v65 > 0x40 )
  {
    if ( v64 )
    {
      v57 = result;
      j_j___libc_free_0_0(v64);
      return v57;
    }
  }
  return result;
}
