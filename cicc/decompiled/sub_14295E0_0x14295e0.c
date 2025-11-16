// Function: sub_14295E0
// Address: 0x14295e0
//
__int64 __fastcall sub_14295E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int8 v3; // al
  bool v4; // dl
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  int v8; // edx
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // edx
  bool v14; // zf
  __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  __int64 v17; // rdx
  __m128i v18; // xmm4
  __m128i v19; // xmm5
  __m128i v20; // xmm6
  __m128i v21; // xmm7
  __m128i v22; // xmm2
  __m128i v23; // xmm3
  __int64 v24; // rdx
  __int64 v25; // rdx
  int v26; // edx
  int v27; // edx
  __int64 v28; // rcx
  unsigned __int64 v29; // rdx
  __int64 v30; // rdx
  int v31; // edx
  __int64 v32; // [rsp+8h] [rbp-D8h]
  __int64 v33; // [rsp+8h] [rbp-D8h]
  __int64 v34; // [rsp+8h] [rbp-D8h]
  __int64 v35; // [rsp+8h] [rbp-D8h]
  __int64 v36; // [rsp+8h] [rbp-D8h]
  __int64 v37; // [rsp+8h] [rbp-D8h]
  __m128i v38; // [rsp+10h] [rbp-D0h] BYREF
  __m128i v39; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+30h] [rbp-B0h]
  __m128i v41; // [rsp+40h] [rbp-A0h]
  __m128i v42; // [rsp+50h] [rbp-90h]
  __int64 v43; // [rsp+60h] [rbp-80h]
  bool v44[8]; // [rsp+70h] [rbp-70h] BYREF
  __m128i v45; // [rsp+78h] [rbp-68h]
  __m128i v46; // [rsp+88h] [rbp-58h]
  __int64 v47; // [rsp+98h] [rbp-48h]
  __int64 v48; // [rsp+A0h] [rbp-40h]
  __int64 v49; // [rsp+A8h] [rbp-38h]
  __int16 v50; // [rsp+B0h] [rbp-30h]

  v2 = *(_QWORD *)(a2 + 72);
  v3 = *(_BYTE *)(v2 + 16);
  if ( v3 > 0x17u && (v3 == 78 || v3 == 29) )
  {
    v45.m128i_i64[0] = 0;
    v45.m128i_i64[1] = -1;
    v4 = (v2 & 0xFFFFFFFFFFFFFFF8LL) != 0;
    v46 = 0u;
    v44[0] = v4;
    v47 = 0;
    v48 = v2;
    v49 = a2;
    v50 = 257;
    if ( (v2 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      goto LABEL_7;
  }
  else
  {
    v44[0] = 0;
    v4 = 0;
    v45.m128i_i64[0] = 0;
    v45.m128i_i64[1] = -1;
    v46 = 0u;
    v47 = 0;
    v48 = v2;
    v49 = a2;
    v50 = 257;
  }
  switch ( *(_BYTE *)(v2 + 16) )
  {
    case '6':
      sub_141EB40(&v38, (__int64 *)v2);
      goto LABEL_25;
    case '7':
      sub_141EDF0(&v38, v2);
      v4 = v44[0];
      v22 = _mm_loadu_si128(&v38);
      v23 = _mm_loadu_si128(&v39);
      v43 = v40;
      v41 = v22;
      v42 = v23;
      break;
    case ':':
      sub_141F110(&v38, v2);
      v4 = v44[0];
      v20 = _mm_loadu_si128(&v38);
      v21 = _mm_loadu_si128(&v39);
      v43 = v40;
      v41 = v20;
      v42 = v21;
      break;
    case ';':
      sub_141F3C0(&v38, v2);
LABEL_25:
      v9 = _mm_loadu_si128(&v38);
      v10 = _mm_loadu_si128(&v39);
      v4 = v44[0];
      v43 = v40;
      v41 = v9;
      v42 = v10;
      break;
    case 'R':
      sub_141F0A0(&v38, v2);
      v4 = v44[0];
      v18 = _mm_loadu_si128(&v38);
      v19 = _mm_loadu_si128(&v39);
      v43 = v40;
      v41 = v18;
      v42 = v19;
      break;
    default:
      break;
  }
  v45 = v41;
  v46 = v42;
  v47 = v43;
  if ( !v4 )
  {
    v11 = (unsigned int)*(unsigned __int8 *)(v2 + 16) - 24;
    if ( (unsigned int)v11 <= 0x36 )
    {
      v12 = 0x44000200000220LL;
      if ( _bittest64(&v12, v11) )
        return a2;
    }
  }
LABEL_7:
  if ( (unsigned __int8)sub_1420060(**(_QWORD **)(a1 + 8), v2) )
  {
    result = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 120LL);
    if ( *(_BYTE *)(a2 + 16) == 22 )
    {
      v24 = *(_QWORD *)(a2 + 112);
      if ( result != v24 )
      {
        if ( v24 != -8 && v24 != 0 && v24 != -16 )
        {
          v34 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 120LL);
          sub_1649B30(a2 + 96);
          result = v34;
        }
        *(_QWORD *)(a2 + 112) = result;
        if ( result != -8 && result != 0 && result != -16 )
        {
          v35 = result;
          sub_164C220(a2 + 96);
          result = v35;
        }
      }
      v25 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v25 + 16) != 22 )
        goto LABEL_65;
      goto LABEL_53;
    }
    if ( *(_BYTE *)(result + 16) == 22 )
      v27 = *(_DWORD *)(result + 84);
    else
      v27 = *(_DWORD *)(result + 72);
    v14 = *(_QWORD *)(a2 - 24) == 0;
    *(_DWORD *)(a2 + 84) = v27;
    if ( v14 )
    {
LABEL_73:
      *(_QWORD *)(a2 - 24) = result;
      v30 = *(_QWORD *)(result + 8);
      *(_QWORD *)(a2 - 16) = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 16) = (a2 - 16) | *(_QWORD *)(v30 + 16) & 3LL;
      *(_QWORD *)(a2 - 8) = *(_QWORD *)(a2 - 8) & 3LL | (result + 8);
      *(_QWORD *)(result + 8) = a2 - 24;
      goto LABEL_55;
    }
LABEL_71:
    v28 = *(_QWORD *)(a2 - 16);
    v29 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v29 = v28;
    if ( v28 )
      *(_QWORD *)(v28 + 16) = *(_QWORD *)(v28 + 16) & 3LL | v29;
    goto LABEL_73;
  }
  result = *(_QWORD *)(a2 - 24);
  if ( result == *(_QWORD *)(*(_QWORD *)(a1 + 8) + 120LL) )
  {
    if ( *(_BYTE *)(a2 + 16) == 22 )
    {
      v25 = *(_QWORD *)(a2 + 112);
      if ( result != v25 )
      {
        if ( v25 != -8 && v25 != 0 && v25 != -16 )
        {
          v36 = *(_QWORD *)(a2 - 24);
          sub_1649B30(a2 + 96);
          result = v36;
        }
        *(_QWORD *)(a2 + 112) = result;
        if ( result == -8 || result == 0 || result == -16 )
        {
          v25 = *(_QWORD *)(a2 - 24);
        }
        else
        {
          v37 = result;
          sub_164C220(a2 + 96);
          v25 = *(_QWORD *)(a2 - 24);
          result = v37;
        }
      }
      if ( *(_BYTE *)(v25 + 16) != 22 )
      {
LABEL_65:
        v26 = *(_DWORD *)(v25 + 72);
        goto LABEL_54;
      }
LABEL_53:
      v26 = *(_DWORD *)(v25 + 84);
LABEL_54:
      *(_DWORD *)(a2 + 88) = v26;
      goto LABEL_55;
    }
    if ( *(_BYTE *)(result + 16) == 22 )
      v31 = *(_DWORD *)(result + 84);
    else
      v31 = *(_DWORD *)(result + 72);
    *(_DWORD *)(a2 + 84) = v31;
    goto LABEL_71;
  }
  result = sub_14285A0(a1 + 16, *(_QWORD *)(a2 - 24), (__int64)v44);
  if ( *(_BYTE *)(a2 + 16) == 22 )
  {
    v6 = *(_QWORD *)(a2 + 112);
    if ( result != v6 )
    {
      if ( v6 != -8 && v6 != 0 && v6 != -16 )
      {
        v32 = result;
        sub_1649B30(a2 + 96);
        result = v32;
      }
      *(_QWORD *)(a2 + 112) = result;
      if ( result != -8 && result != 0 && result != -16 )
      {
        v33 = result;
        sub_164C220(a2 + 96);
        result = v33;
      }
    }
    v7 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v7 + 16) == 22 )
      v8 = *(_DWORD *)(v7 + 84);
    else
      v8 = *(_DWORD *)(v7 + 72);
    *(_DWORD *)(a2 + 88) = v8;
  }
  else
  {
    if ( *(_BYTE *)(result + 16) == 22 )
      v13 = *(_DWORD *)(result + 84);
    else
      v13 = *(_DWORD *)(result + 72);
    v14 = *(_QWORD *)(a2 - 24) == 0;
    *(_DWORD *)(a2 + 84) = v13;
    if ( !v14 )
    {
      v15 = *(_QWORD *)(a2 - 16);
      v16 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v16 = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
    }
    *(_QWORD *)(a2 - 24) = result;
    v17 = *(_QWORD *)(result + 8);
    *(_QWORD *)(a2 - 16) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = (a2 - 16) | *(_QWORD *)(v17 + 16) & 3LL;
    *(_QWORD *)(a2 - 8) = *(_QWORD *)(a2 - 8) & 3LL | (result + 8);
    *(_QWORD *)(result + 8) = a2 - 24;
  }
  if ( result != *(_QWORD *)(*(_QWORD *)(a1 + 8) + 120LL) )
  {
    if ( HIBYTE(v50) && (_BYTE)v50 == 3 )
    {
      if ( *(_BYTE *)(a2 + 81) )
        *(_BYTE *)(a2 + 80) = 3;
      else
        *(_WORD *)(a2 + 80) = 259;
    }
    return result;
  }
LABEL_55:
  if ( *(_BYTE *)(a2 + 81) )
    *(_BYTE *)(a2 + 81) = 0;
  return result;
}
