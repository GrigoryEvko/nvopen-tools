// Function: sub_894C10
// Address: 0x894c10
//
__int64 __fastcall sub_894C10(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  __int64 *v4; // r14
  __int64 v5; // rdx
  __m128i *v8; // rsi
  __int64 v9; // r15
  char v10; // al
  __int64 v11; // r10
  __int64 v12; // rcx
  __int64 v13; // rdi
  const __m128i *v14; // rcx
  int v15; // eax
  __m128i v16; // xmm1
  char v17; // r9
  __int64 v18; // r10
  __int64 v19; // rdi
  __int64 v20; // r10
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 *v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 *v32; // r9
  __int64 v33; // r8
  __int64 v34; // rdx
  __m128i *v35; // rax
  char v36; // r12
  __int64 v37; // rax
  __int64 v38; // [rsp+8h] [rbp-58h]
  __int64 v39; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+28h] [rbp-38h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  const __m128i *v45; // [rsp+28h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 88);
  result = *(_QWORD *)(v2 + 152);
  if ( *(_BYTE *)(result + 140) != 7 )
    return result;
  v4 = *(__int64 **)(a2 + 96);
  v5 = *(_QWORD *)(result + 168);
  v8 = *(__m128i **)(v5 + 56);
  if ( !v4 )
  {
    v43 = a2;
    v9 = 0;
    v11 = 0;
    goto LABEL_22;
  }
  v9 = v4[4];
  v43 = a2;
  v10 = *(_BYTE *)(v9 + 80);
  if ( v10 == 20 )
  {
    v11 = *(_QWORD *)(v9 + 88);
    if ( (*(_BYTE *)(v2 + 206) & 2) != 0 )
      goto LABEL_11;
    v12 = *(_QWORD *)(v11 + 88);
    if ( !v12 || (*(_BYTE *)(v11 + 160) & 1) != 0 )
    {
      v43 = v4[4];
      goto LABEL_11;
    }
    v43 = v4[4];
    v10 = *(_BYTE *)(v12 + 80);
    v9 = *(_QWORD *)(v11 + 88);
  }
  switch ( v10 )
  {
    case 4:
    case 5:
      v11 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 80LL);
      break;
    case 6:
      v11 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v11 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v11 = *(_QWORD *)(v9 + 88);
      break;
    default:
      BUG();
  }
LABEL_11:
  result = *(_QWORD *)(v11 + 176);
  if ( dword_4D047B0 )
  {
    v13 = *(_QWORD *)result;
    if ( (*(_BYTE *)(v11 + 424) & 4) == 0 && a2 != v13 )
    {
      v38 = v5;
      v39 = *(_QWORD *)(v11 + 176);
      v41 = v11;
      sub_894C00(v13);
      v5 = v38;
      result = v39;
      v11 = v41;
    }
  }
  *(_BYTE *)(v11 + 424) |= 4u;
  if ( (*(_BYTE *)(v2 + 195) & 8) != 0 && v2 != result )
  {
    result = *(_QWORD *)(*(_QWORD *)(result + 152) + 168LL);
    v14 = *(const __m128i **)(result + 56);
    if ( v14 )
    {
      if ( !v8 )
      {
        v42 = v5;
        v45 = *(const __m128i **)(result + 56);
        v35 = sub_725E60();
        v14 = v45;
        v8 = v35;
        *(_QWORD *)(v42 + 56) = v35;
      }
      *v8 = _mm_loadu_si128(v14);
      v15 = v8->m128i_u8[0];
      v16 = _mm_loadu_si128(v14 + 1);
      v8->m128i_i64[1] = v2;
      v8[1] = v16;
      result = v15 & 0xFFFFFF9F | 0x40;
      v8->m128i_i8[0] = result;
      return result;
    }
  }
LABEL_22:
  if ( !v8 || (v8->m128i_i8[0] & 0x20) == 0 )
    return result;
  if ( v11 )
  {
    result = sub_890B60(v4[4]);
    if ( (_DWORD)result )
      return result;
    v8->m128i_i64[1] = 0;
    v8->m128i_i8[0] = v17 & 0xDF;
    v19 = *(_QWORD *)(v18 + 368);
    v40 = v18;
    if ( !v19 )
      return result;
    sub_864700(v19, 0, v2, a2, v9, *(_QWORD *)(v2 + 240), 1, (*(_BYTE *)(v2 + 195) & 8) == 0 ? 0x20000 : 131074);
    if ( (*(_BYTE *)(a2 + 81) & 0x10) != 0 )
    {
      v33 = *(_QWORD *)(v2 + 152);
      v34 = **(_QWORD **)(a2 + 64);
      if ( (unsigned __int8)(*(_BYTE *)(v34 + 80) - 4) <= 1u && *(_QWORD *)(*(_QWORD *)(v34 + 96) + 72LL) )
      {
        v36 = *(_BYTE *)(*(_QWORD *)(v34 + 88) + 178LL);
        sub_8600D0(1u, -1, v33, 0);
        v20 = v40;
        if ( (v36 & 1) == 0 )
          goto LABEL_49;
      }
      else
      {
        sub_8600D0(1u, -1, v33, 0);
        v20 = v40;
      }
    }
    else
    {
      sub_8600D0(1u, -1, *(_QWORD *)(v2 + 152), 0);
      v20 = v40;
    }
    if ( (*(_BYTE *)(v2 + 195) & 8) == 0 )
    {
      v21 = qword_4F04C68[0] + 776LL * dword_4F04C64;
LABEL_31:
      *(_BYTE *)(v21 + 11) |= 0x40u;
      if ( a1 )
      {
        v22 = *(_QWORD *)(a1 + 440);
        v23 = *(_QWORD *)(a1 + 448);
      }
      else
      {
        v22 = v4[8];
        v23 = v4[13];
      }
      if ( v22 )
      {
        v44 = v20;
        sub_886000(v22);
        v20 = v44;
        v21 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      }
      *(_QWORD *)(v21 + 752) = v23;
      v24 = v20 + 336;
      sub_625150(v2, v20 + 336, 0);
      sub_863FC0(v2, v24, v25, v26, v27, v28);
      sub_863FE0(v2, v24, v29, v30, v31, v32);
      result = (__int64)&dword_4F06978;
      if ( !(dword_4F06978 | dword_4D048B8) )
      {
        result = *(_QWORD *)(*(_QWORD *)(v2 + 152) + 168LL);
        *(_QWORD *)(result + 56) = 0;
      }
      return result;
    }
LABEL_49:
    v37 = 776LL * dword_4F04C64;
    *(_BYTE *)(qword_4F04C68[0] + v37 + 6) |= 0x20u;
    v21 = qword_4F04C68[0] + v37;
    *(_DWORD *)(v21 + 420) = *(_DWORD *)(v43 + 44) - 1;
    goto LABEL_31;
  }
  if ( (*(_BYTE *)(v2 + 89) & 4) != 0 && (*(_BYTE *)(v2 + 195) & 8) == 0 )
    return sub_5EA8F0(v2, (__int64)v8);
  if ( (*(_BYTE *)(a2 + 104) & 2) != 0 )
    return sub_625480(a2, (unsigned __int64)v8);
  return result;
}
