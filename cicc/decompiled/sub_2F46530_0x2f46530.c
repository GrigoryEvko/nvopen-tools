// Function: sub_2F46530
// Address: 0x2f46530
//
__int64 __fastcall sub_2F46530(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v11; // rax
  unsigned int v12; // ecx
  _WORD *v13; // r8
  unsigned int v14; // eax
  __int64 v15; // rdi
  __m128i *v16; // r13
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  unsigned int v21; // ecx
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 v25; // r9
  unsigned int v26; // edi
  _DWORD *v27; // rdx
  unsigned int v28; // r8d
  __m128i *v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rcx
  const __m128i *v33; // rdx
  __m128i *v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // r8
  __int64 v37; // rdi
  const void *v38; // rsi
  unsigned int *v39; // rax
  int v40; // edi
  int v41; // edi
  int v42; // r8d
  int v43; // r8d
  __int64 v44; // r9
  unsigned int v45; // edx
  unsigned int v46; // r11d
  int v47; // esi
  unsigned int *v48; // rcx
  int v49; // r9d
  int v50; // r9d
  int v51; // esi
  unsigned int *v52; // rdx
  __int64 v53; // r10
  unsigned int v54; // ecx
  unsigned int v55; // r8d
  int v56; // [rsp+4h] [rbp-5Ch]
  __int64 v57; // [rsp+8h] [rbp-58h]
  int v58; // [rsp+8h] [rbp-58h]
  __int64 v59; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v60; // [rsp+18h] [rbp-48h]
  int v61; // [rsp+1Ch] [rbp-44h]
  char v62; // [rsp+20h] [rbp-40h]

  if ( *(_QWORD *)(a1 + 368) )
  {
    LODWORD(v59) = a4;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD, __int64 *))(a1 + 376))(
            a1 + 352,
            *(_QWORD *)(a1 + 16),
            *(_QWORD *)(a1 + 8),
            &v59) )
      return 0;
  }
  v11 = *(_QWORD *)(a1 + 624);
  v60 = a4;
  v62 = 0;
  v12 = *(_DWORD *)(a1 + 424);
  v59 = 0;
  v13 = (_WORD *)(v11 + 2LL * (a4 & 0x7FFFFFFF));
  v61 = 0;
  v14 = (unsigned __int16)*v13;
  if ( v14 >= v12 )
    goto LABEL_25;
  v15 = *(_QWORD *)(a1 + 416);
  while ( 1 )
  {
    v16 = (__m128i *)(v15 + 24LL * v14);
    if ( (a4 & 0x7FFFFFFF) == (v16->m128i_i32[2] & 0x7FFFFFFF) )
      break;
    v14 += 0x10000;
    if ( v12 <= v14 )
      goto LABEL_25;
  }
  if ( v16 == (__m128i *)(v15 + 24LL * v12) )
  {
LABEL_25:
    *v13 = v12;
    v30 = *(unsigned int *)(a1 + 424);
    v31 = v30 + 1;
    if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 428) )
    {
      v36 = *(_QWORD *)(a1 + 416);
      v37 = a1 + 416;
      v38 = (const void *)(a1 + 432);
      if ( v36 > (unsigned __int64)&v59 || (v57 = *(_QWORD *)(a1 + 416), (unsigned __int64)&v59 >= v36 + 24 * v30) )
      {
        sub_C8D5F0(v37, v38, v31, 0x18u, v36, a6);
        v32 = *(_QWORD *)(a1 + 416);
        v30 = *(unsigned int *)(a1 + 424);
        v33 = (const __m128i *)&v59;
      }
      else
      {
        sub_C8D5F0(v37, v38, v31, 0x18u, v36, a6);
        v32 = *(_QWORD *)(a1 + 416);
        v30 = *(unsigned int *)(a1 + 424);
        v33 = (const __m128i *)((char *)&v59 + v32 - v57);
      }
    }
    else
    {
      v32 = *(_QWORD *)(a1 + 416);
      v33 = (const __m128i *)&v59;
    }
    v34 = (__m128i *)(v32 + 24 * v30);
    *v34 = _mm_loadu_si128(v33);
    v34[1].m128i_i64[0] = v33[1].m128i_i64[0];
    v35 = (unsigned int)(*(_DWORD *)(a1 + 424) + 1);
    *(_DWORD *)(a1 + 424) = v35;
    v16 = (__m128i *)(*(_QWORD *)(a1 + 416) + 24 * v35 - 24);
    if ( (((*(_BYTE *)(a3 + 3) & 0x40) != 0) & ((*(_BYTE *)(a3 + 3) >> 4) ^ 1)) == 0 )
    {
      if ( (unsigned __int8)sub_2F462B0((_QWORD *)a1, a4) )
        v16->m128i_i8[14] = 1;
      else
        *(_BYTE *)(a3 + 3) |= 0x40u;
    }
  }
  if ( !v16->m128i_i16[6] )
  {
    v17 = 0;
    if ( *(_WORD *)(a2 + 68) == 20 )
    {
      v23 = *(_QWORD *)(a2 + 32);
      if ( (*(_DWORD *)(v23 + 40) & 0xFFF00) == 0 )
      {
        v17 = *(unsigned int *)(v23 + 8);
        if ( (int)v17 < 0 )
          v17 = 0;
      }
    }
    sub_2F44460(a1, (_BYTE *)a2, (__int64)v16, v17, 0, a6);
  }
  v16->m128i_i64[0] = a2;
  if ( *(_WORD *)(a2 + 68) == 21 )
  {
    v24 = *(_DWORD *)(a1 + 664);
    if ( v24 )
    {
      v25 = *(_QWORD *)(a1 + 648);
      v26 = (v24 - 1) & (37 * a4);
      v27 = (_DWORD *)(v25 + 32LL * v26);
      v28 = *v27;
      if ( *v27 == a4 )
      {
LABEL_23:
        v29 = (__m128i *)(v27 + 2);
LABEL_24:
        *v29 = _mm_loadu_si128(v16);
        v29[1].m128i_i8[0] = v16[1].m128i_i8[0];
        goto LABEL_13;
      }
      v58 = 1;
      v39 = 0;
      v56 = 37 * a4;
      while ( v28 != -1 )
      {
        if ( v28 == -2 && !v39 )
          v39 = v27;
        v26 = (v24 - 1) & (v58 + v26);
        v27 = (_DWORD *)(v25 + 32LL * v26);
        v28 = *v27;
        if ( *v27 == a4 )
          goto LABEL_23;
        ++v58;
      }
      v40 = *(_DWORD *)(a1 + 656);
      if ( !v39 )
        v39 = v27;
      ++*(_QWORD *)(a1 + 640);
      v41 = v40 + 1;
      if ( 4 * v41 < 3 * v24 )
      {
        if ( v24 - *(_DWORD *)(a1 + 660) - v41 > v24 >> 3 )
        {
LABEL_41:
          *(_DWORD *)(a1 + 656) = v41;
          if ( *v39 != -1 )
            --*(_DWORD *)(a1 + 660);
          *v39 = a4;
          v29 = (__m128i *)(v39 + 2);
          v29->m128i_i64[0] = 0;
          v29->m128i_i64[1] = 0;
          v29[1].m128i_i8[0] = 0;
          goto LABEL_24;
        }
        sub_2F42650(a1 + 640, v24);
        v49 = *(_DWORD *)(a1 + 664);
        if ( v49 )
        {
          v50 = v49 - 1;
          v51 = 1;
          v52 = 0;
          v53 = *(_QWORD *)(a1 + 648);
          v54 = v50 & v56;
          v41 = *(_DWORD *)(a1 + 656) + 1;
          v39 = (unsigned int *)(v53 + 32LL * (v50 & (unsigned int)v56));
          v55 = *v39;
          if ( a4 != *v39 )
          {
            while ( v55 != -1 )
            {
              if ( v55 == -2 && !v52 )
                v52 = v39;
              v54 = v50 & (v51 + v54);
              v39 = (unsigned int *)(v53 + 32LL * v54);
              v55 = *v39;
              if ( *v39 == a4 )
                goto LABEL_41;
              ++v51;
            }
            if ( v52 )
              v39 = v52;
          }
          goto LABEL_41;
        }
LABEL_73:
        ++*(_DWORD *)(a1 + 656);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 640);
    }
    sub_2F42650(a1 + 640, 2 * v24);
    v42 = *(_DWORD *)(a1 + 664);
    if ( v42 )
    {
      v43 = v42 - 1;
      v44 = *(_QWORD *)(a1 + 648);
      v45 = v43 & (37 * a4);
      v41 = *(_DWORD *)(a1 + 656) + 1;
      v39 = (unsigned int *)(v44 + 32LL * v45);
      v46 = *v39;
      if ( a4 != *v39 )
      {
        v47 = 1;
        v48 = 0;
        while ( v46 != -1 )
        {
          if ( !v48 && v46 == -2 )
            v48 = v39;
          v45 = v43 & (v47 + v45);
          v39 = (unsigned int *)(v44 + 32LL * v45);
          v46 = *v39;
          if ( *v39 == a4 )
            goto LABEL_41;
          ++v47;
        }
        if ( v48 )
          v39 = v48;
      }
      goto LABEL_41;
    }
    goto LABEL_73;
  }
LABEL_13:
  v18 = *(_QWORD *)(a1 + 16);
  v19 = *(_QWORD *)(v18 + 8);
  v20 = *(_DWORD *)(v19 + 24LL * v16->m128i_u16[6] + 16) >> 12;
  v21 = *(_DWORD *)(v19 + 24LL * v16->m128i_u16[6] + 16) & 0xFFF;
  v22 = *(_QWORD *)(v18 + 56) + 2 * v20;
  do
  {
    if ( !v22 )
      break;
    v22 += 2;
    *(_DWORD *)(*(_QWORD *)(a1 + 1112) + 4LL * v21) = *(_DWORD *)(a1 + 1104) | 1;
    v21 += *(__int16 *)(v22 - 2);
  }
  while ( *(_WORD *)(v22 - 2) );
  return sub_2F41240(a1, a2, a3, (__int64)v16);
}
