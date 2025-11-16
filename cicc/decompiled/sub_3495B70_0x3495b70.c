// Function: sub_3495B70
// Address: 0x3495b70
//
__int64 __fastcall sub_3495B70(
        _WORD *a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int128 a8,
        _QWORD *a9,
        _QWORD *a10)
{
  __int64 v11; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int); // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int128 v21; // rax
  __int64 v22; // r9
  unsigned __int8 *v23; // rax
  __int32 v24; // edx
  __int64 v25; // r9
  __int32 v26; // edx
  __int64 v27; // rax
  __int64 *v28; // rdi
  char v29; // cl
  __int64 (__fastcall *v30)(__int64, unsigned __int8 *); // rbx
  unsigned __int8 *v31; // rax
  char v32; // al
  __m128i v33; // xmm0
  __m128i v34; // xmm1
  __int64 v35; // rbx
  bool v36; // zf
  __int64 v37; // rax
  _QWORD *v38; // rsi
  __int64 v39; // rax
  _QWORD *v40; // rbx
  __int64 result; // rax
  unsigned __int8 *v42; // rax
  __int32 v43; // edx
  __int32 v44; // edx
  __int64 v45; // rax
  __m128i v46; // xmm2
  __m128i v47; // xmm3
  __int64 (__fastcall *v48)(__int64, __int64, unsigned int); // [rsp+18h] [rbp-138h]
  unsigned int v49; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v50; // [rsp+28h] [rbp-128h]
  __int128 v51; // [rsp+30h] [rbp-120h]
  __int32 v52; // [rsp+30h] [rbp-120h]
  unsigned __int8 *v53; // [rsp+40h] [rbp-110h]
  __int32 v54; // [rsp+48h] [rbp-108h]
  int v55; // [rsp+4Ch] [rbp-104h]
  __m128i v56; // [rsp+50h] [rbp-100h] BYREF
  unsigned int v57; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v58; // [rsp+68h] [rbp-E8h]
  __int64 v59; // [rsp+70h] [rbp-E0h]
  __int64 v60; // [rsp+78h] [rbp-D8h]
  __int64 v61; // [rsp+80h] [rbp-D0h]
  __int64 v62; // [rsp+88h] [rbp-C8h]
  __int64 v63[4]; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v64; // [rsp+B0h] [rbp-A0h]
  __int64 v65; // [rsp+B8h] [rbp-98h]
  __int64 v66; // [rsp+C0h] [rbp-90h]
  __int64 v67; // [rsp+C8h] [rbp-88h]
  __int64 v68; // [rsp+D0h] [rbp-80h]
  __m128i v69; // [rsp+E0h] [rbp-70h] BYREF
  __m128i v70; // [rsp+F0h] [rbp-60h]
  __m128i v71; // [rsp+100h] [rbp-50h]
  __m128i v72; // [rsp+110h] [rbp-40h]

  v11 = a3;
  v13 = *(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6;
  v56.m128i_i64[0] = a5;
  LOWORD(a3) = *(_WORD *)v13;
  v14 = *(_QWORD *)(v13 + 8);
  v56.m128i_i64[1] = a6;
  LOWORD(v57) = a3;
  v58 = v14;
  if ( (_WORD)a3 )
  {
    if ( (_WORD)a3 == 1 || (unsigned __int16)(a3 - 504) <= 7u )
      goto LABEL_44;
    v15 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a3 - 16];
    LOBYTE(a3) = byte_444C4A0[16 * (unsigned __int16)a3 - 8];
  }
  else
  {
    v15 = sub_3007260((__int64)&v57);
    v59 = v15;
    v60 = a3;
  }
  v69.m128i_i8[8] = a3;
  v69.m128i_i64[0] = 2 * v15;
  v16 = sub_CA1930(&v69);
  switch ( v16 )
  {
    case 1u:
    case 2u:
    case 4u:
    case 8u:
      return sub_3469040(a7, (__int64)a1, (_QWORD *)a2, v11, a4, (__int64)a9, (__int64)a10, *(_OWORD *)&v56, a8, 0, 0);
    case 0x10u:
      v49 = 6;
      v48 = 0;
      goto LABEL_12;
    case 0x20u:
      v49 = 7;
      v18 = 14;
      v48 = 0;
      v55 = 14;
      break;
    case 0x40u:
      v49 = 8;
      v18 = 15;
      v48 = 0;
      v55 = 15;
      break;
    case 0x80u:
      v49 = 9;
      v48 = 0;
LABEL_35:
      v55 = 16;
      v18 = 16;
      break;
    default:
      v49 = sub_3007020(*(_QWORD **)(a2 + 64), v16);
      v48 = v17;
      if ( (_WORD)v49 == 6 )
      {
LABEL_12:
        v55 = 13;
        v18 = 13;
        break;
      }
      v55 = 14;
      v18 = 14;
      if ( (_WORD)v49 != 7 )
      {
        v55 = 15;
        v18 = 15;
        if ( (_WORD)v49 != 8 )
        {
          if ( (_WORD)v49 != 9 )
            return sub_3469040(
                     a7,
                     (__int64)a1,
                     (_QWORD *)a2,
                     v11,
                     a4,
                     (__int64)a9,
                     (__int64)a10,
                     *(_OWORD *)&v56,
                     a8,
                     0,
                     0);
          goto LABEL_35;
        }
      }
      break;
  }
  if ( !*(_QWORD *)&a1[4 * v18 + 262644] )
    return sub_3469040(a7, (__int64)a1, (_QWORD *)a2, v11, a4, (__int64)a9, (__int64)a10, *(_OWORD *)&v56, a8, 0, 0);
  if ( !a4 )
  {
    v42 = sub_3400BD0(a2, 0, v11, v57, v58, 0, a7, 0);
    v54 = v43;
    v53 = v42;
    v50 = sub_3400BD0(a2, 0, v11, v57, v58, 0, a7, 0);
    v52 = v44;
    goto LABEL_18;
  }
  if ( (_WORD)v57 )
  {
    if ( (_WORD)v57 != 1 && (unsigned __int16)(v57 - 504) > 7u )
    {
      v19 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v57 - 16];
      goto LABEL_17;
    }
LABEL_44:
    BUG();
  }
  v19 = sub_3007260((__int64)&v57);
  v61 = v19;
  v62 = v20;
LABEL_17:
  *(_QWORD *)&v21 = sub_3400E40(a2, (unsigned int)(v19 - 1), v57, v58, v11, a7);
  v51 = v21;
  v23 = sub_3406EB0((_QWORD *)a2, 0xBFu, v11, v57, v58, v22, *(_OWORD *)&v56, v21);
  v54 = v24;
  v53 = v23;
  v50 = sub_3406EB0((_QWORD *)a2, 0xBFu, v11, v57, v58, v25, a8, v51);
  v52 = v26;
LABEL_18:
  v27 = *(_QWORD *)a1;
  v28 = *(__int64 **)(a2 + 40);
  v64 = 0;
  v65 = 0;
  v29 = a4 & 1 | 0xC;
  v30 = *(__int64 (__fastcall **)(__int64, unsigned __int8 *))(v27 + 2376);
  v66 = 0;
  v67 = 0;
  LOBYTE(v68) = v29;
  v31 = (unsigned __int8 *)sub_2E79000(v28);
  if ( v30 == sub_302E250 )
    v32 = *v31 ^ 1;
  else
    v32 = v30((__int64)a1, v31);
  if ( v32 )
  {
    v33 = _mm_load_si128(&v56);
    v34 = _mm_loadu_si128((const __m128i *)&a8);
    v70.m128i_i64[0] = (__int64)v53;
    v69 = v33;
    v70.m128i_i32[2] = v54;
    v71 = v34;
    v72.m128i_i64[0] = (__int64)v50;
    v72.m128i_i32[2] = v52;
  }
  else
  {
    v46 = _mm_load_si128(&v56);
    v47 = _mm_loadu_si128((const __m128i *)&a8);
    v69.m128i_i64[0] = (__int64)v53;
    v70 = v46;
    v69.m128i_i32[2] = v54;
    v72 = v47;
    v71.m128i_i64[0] = (__int64)v50;
    v71.m128i_i32[2] = v52;
  }
  sub_3494590((__int64)v63, a1, a2, v55, v49, v48, (__int64)&v69, 4u, v64, v65, v66, v67, v68, v11, 0, 0);
  v35 = v63[0];
  v36 = *(_BYTE *)sub_2E79000(*(__int64 **)(a2 + 40)) == 0;
  v37 = *(_QWORD *)(v35 + 40);
  v38 = a9;
  if ( v36 )
  {
    *a9 = *(_QWORD *)v37;
    *((_DWORD *)v38 + 2) = *(_DWORD *)(v37 + 8);
    v39 = *(_QWORD *)(v35 + 40);
    v40 = a10;
    *a10 = *(_QWORD *)(v39 + 40);
    result = *(unsigned int *)(v39 + 48);
  }
  else
  {
    *a9 = *(_QWORD *)(v37 + 40);
    *((_DWORD *)v38 + 2) = *(_DWORD *)(v37 + 48);
    v45 = *(_QWORD *)(v35 + 40);
    v40 = a10;
    *a10 = *(_QWORD *)v45;
    result = *(unsigned int *)(v45 + 8);
  }
  *((_DWORD *)v40 + 2) = result;
  return result;
}
