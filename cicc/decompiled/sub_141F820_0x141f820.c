// Function: sub_141F820
// Address: 0x141f820
//
__m128i *__fastcall sub_141F820(__m128i *a1, __int64 a2, unsigned int a3, _QWORD *a4)
{
  unsigned __int64 v6; // rbx
  __int64 v7; // r14
  int v8; // eax
  __int64 v9; // r12
  _QWORD *v10; // rax
  unsigned __int64 v11; // rbx
  __int64 v12; // rax
  __m128i v13; // xmm0
  __int64 v14; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __m128i v19; // xmm7
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rax
  __m128i v27; // xmm5
  __int64 v28; // rax
  __int64 v29; // rdx
  _QWORD *v30; // rax
  __m128i v31; // xmm1
  __int64 v32; // rax
  __int64 v33; // rdx
  _QWORD *v34; // rax
  __m128i v35; // xmm2
  __int64 v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rbx
  __int64 v39; // rsi
  __int64 v40; // rsi
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rbx
  __m128i v44; // xmm3
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rbx
  __m128i v48; // xmm4
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r14
  unsigned __int64 v52; // r13
  __int64 v53; // rax
  __int64 v54; // r14
  unsigned __int64 v55; // r13
  __m128i v56; // xmm6
  __int64 v57; // rax
  char v58; // [rsp+0h] [rbp-70h]
  __int64 v60; // [rsp+8h] [rbp-68h]
  __int64 v61; // [rsp+8h] [rbp-68h]
  int v62; // [rsp+1Ch] [rbp-54h] BYREF
  __m128i v63; // [rsp+20h] [rbp-50h] BYREF
  __int64 v64; // [rsp+30h] [rbp-40h]

  v6 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v63 = 0u;
  v64 = 0;
  sub_14A8180(a2 & 0xFFFFFFFFFFFFFFF8LL, &v63, 0);
  v7 = 24LL * a3;
  v58 = (a2 >> 2) & 1;
  v8 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20);
  if ( v58 )
  {
    v9 = *(_QWORD *)(v6 + v7 - 24LL * (v8 & 0xFFFFFFF));
    if ( *(_BYTE *)(v6 + 16) != 78 )
    {
LABEL_3:
      v10 = (_QWORD *)(v6 - 24);
LABEL_4:
      if ( !*(_BYTE *)(*v10 + 16LL)
        && (unsigned __int8)sub_149CB50(*a4, *v10, &v62)
        && v62 == 295
        && (*(_BYTE *)(*a4 + 73LL) & 0xC0) != 0 )
      {
        if ( a3 == 1 )
        {
          v56 = _mm_loadu_si128(&v63);
          v57 = v64;
          a1->m128i_i64[0] = v9;
          a1->m128i_i64[1] = 16;
          a1[2].m128i_i64[0] = v57;
          a1[1] = v56;
          return a1;
        }
        v11 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        v17 = *(_QWORD *)(v11 + 48);
        if ( *(_BYTE *)(v17 + 16) == 13 )
        {
          if ( *(_DWORD *)(v17 + 32) <= 0x40u )
            v18 = *(_QWORD *)(v17 + 24);
          else
            v18 = **(_QWORD **)(v17 + 24);
          a1->m128i_i64[1] = v18;
          v19 = _mm_loadu_si128(&v63);
          v20 = v64;
          a1->m128i_i64[0] = v9;
          a1[1] = v19;
          a1[2].m128i_i64[0] = v20;
          return a1;
        }
      }
      else
      {
        v11 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
      }
      v12 = *(_QWORD *)(v11 + 24LL * a3);
      v13 = _mm_loadu_si128(&v63);
      a1->m128i_i64[1] = -1;
      a1->m128i_i64[0] = v12;
      v14 = v64;
      a1[1] = v13;
      a1[2].m128i_i64[0] = v14;
      return a1;
    }
  }
  else
  {
    v9 = *(_QWORD *)(v6 + v7 - 24LL * (v8 & 0xFFFFFFF));
    if ( *(_BYTE *)(v6 + 16) != 78 )
      goto LABEL_15;
  }
  v16 = *(_QWORD *)(v6 - 24);
  if ( *(_BYTE *)(v16 + 16) || (*(_BYTE *)(v16 + 33) & 0x20) == 0 || !v6 )
  {
LABEL_14:
    if ( v58 )
      goto LABEL_3;
LABEL_15:
    v10 = (_QWORD *)(v6 - 72);
    goto LABEL_4;
  }
  v21 = sub_15F2050(v6);
  v22 = sub_1632FA0(v21);
  v23 = *(_QWORD *)(v6 - 24);
  if ( *(_BYTE *)(v23 + 16) )
LABEL_69:
    BUG();
  v24 = *(_DWORD *)(v23 + 36);
  if ( v24 > 0x89 )
  {
    if ( v24 != 1079 )
    {
      if ( v24 == 1154 )
      {
        v37 = *(__int64 **)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
        v38 = 1;
        v39 = *v37;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v39 + 8) )
          {
            case 1:
              v46 = 16;
              goto LABEL_46;
            case 2:
              v46 = 32;
              goto LABEL_46;
            case 3:
            case 9:
              v46 = 64;
              goto LABEL_46;
            case 4:
              v46 = 80;
              goto LABEL_46;
            case 5:
            case 6:
              v46 = 128;
              goto LABEL_46;
            case 7:
              v46 = 8 * (unsigned int)sub_15A9520(v22, 0);
              goto LABEL_46;
            case 0xB:
              v46 = *(_DWORD *)(v39 + 8) >> 8;
              goto LABEL_46;
            case 0xD:
              v46 = 8LL * *(_QWORD *)sub_15A9930(v22, v39);
              goto LABEL_46;
            case 0xE:
              v54 = *(_QWORD *)(v39 + 24);
              v61 = *(_QWORD *)(v39 + 32);
              v55 = (unsigned int)sub_15A9FE0(v22, v54);
              v46 = 8 * v55 * v61 * ((v55 + ((unsigned __int64)(sub_127FA20(v22, v54) + 7) >> 3) - 1) / v55);
              goto LABEL_46;
            case 0xF:
              v46 = 8 * (unsigned int)sub_15A9520(v22, *(_DWORD *)(v39 + 8) >> 8);
LABEL_46:
              v47 = v46 * v38;
              v48 = _mm_loadu_si128(&v63);
              v49 = v64;
              a1->m128i_i64[0] = v9;
              a1[2].m128i_i64[0] = v49;
              a1[1] = v48;
              a1->m128i_i64[1] = (unsigned __int64)(v47 + 7) >> 3;
              return a1;
            case 0x10:
              v53 = *(_QWORD *)(v39 + 32);
              v39 = *(_QWORD *)(v39 + 24);
              v38 *= v53;
              continue;
            default:
              goto LABEL_69;
          }
        }
      }
      goto LABEL_14;
    }
    v40 = *(_QWORD *)v6;
    v41 = 1;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v40 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v50 = *(_QWORD *)(v40 + 32);
          v40 = *(_QWORD *)(v40 + 24);
          v41 *= v50;
          continue;
        case 1:
          v42 = 16;
          break;
        case 2:
          v42 = 32;
          break;
        case 3:
        case 9:
          v42 = 64;
          break;
        case 4:
          v42 = 80;
          break;
        case 5:
        case 6:
          v42 = 128;
          break;
        case 7:
          v42 = 8 * (unsigned int)sub_15A9520(v22, 0);
          break;
        case 0xB:
          v42 = *(_DWORD *)(v40 + 8) >> 8;
          break;
        case 0xD:
          v42 = 8LL * *(_QWORD *)sub_15A9930(v22, v40);
          break;
        case 0xE:
          v51 = *(_QWORD *)(v40 + 24);
          v60 = *(_QWORD *)(v40 + 32);
          v52 = (unsigned int)sub_15A9FE0(v22, v51);
          v42 = 8 * v52 * v60 * ((v52 + ((unsigned __int64)(sub_127FA20(v22, v51) + 7) >> 3) - 1) / v52);
          break;
        case 0xF:
          v42 = 8 * (unsigned int)sub_15A9520(v22, *(_DWORD *)(v40 + 8) >> 8);
          break;
      }
      break;
    }
    v43 = v42 * v41;
    v44 = _mm_loadu_si128(&v63);
    v45 = v64;
    a1->m128i_i64[0] = v9;
    a1[2].m128i_i64[0] = v45;
    a1[1] = v44;
    a1->m128i_i64[1] = (unsigned __int64)(v43 + 7) >> 3;
  }
  else
  {
    if ( v24 <= 0x70 )
      goto LABEL_14;
    switch ( v24 )
    {
      case 0x71u:
        v33 = *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
        v34 = *(_QWORD **)(v33 + 24);
        if ( *(_DWORD *)(v33 + 32) > 0x40u )
          v34 = (_QWORD *)*v34;
        a1->m128i_i64[1] = (__int64)v34;
        v35 = _mm_loadu_si128(&v63);
        v36 = v64;
        a1->m128i_i64[0] = v9;
        a1[1] = v35;
        a1[2].m128i_i64[0] = v36;
        break;
      case 0x72u:
      case 0x74u:
      case 0x75u:
        v29 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        v30 = *(_QWORD **)(v29 + 24);
        if ( *(_DWORD *)(v29 + 32) > 0x40u )
          v30 = (_QWORD *)*v30;
        a1->m128i_i64[1] = (__int64)v30;
        v31 = _mm_loadu_si128(&v63);
        v32 = v64;
        a1->m128i_i64[0] = v9;
        a1[1] = v31;
        a1[2].m128i_i64[0] = v32;
        break;
      case 0x85u:
      case 0x87u:
      case 0x89u:
        v25 = *(_QWORD *)(v6 + 24 * (2LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v25 + 16) != 13 )
          goto LABEL_14;
        if ( *(_DWORD *)(v25 + 32) <= 0x40u )
          v26 = *(_QWORD *)(v25 + 24);
        else
          v26 = **(_QWORD **)(v25 + 24);
        a1->m128i_i64[1] = v26;
        v27 = _mm_loadu_si128(&v63);
        v28 = v64;
        a1->m128i_i64[0] = v9;
        a1[1] = v27;
        a1[2].m128i_i64[0] = v28;
        break;
      default:
        goto LABEL_14;
    }
  }
  return a1;
}
