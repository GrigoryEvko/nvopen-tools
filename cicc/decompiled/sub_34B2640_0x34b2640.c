// Function: sub_34B2640
// Address: 0x34b2640
//
__int64 __fastcall sub_34B2640(unsigned __int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rax
  _BYTE *v4; // rdx
  __int64 result; // rax
  unsigned __int64 v6; // r13
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  bool v11; // zf
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a2;
  v3 = *(_BYTE **)(a2 + 24);
  v4 = *(_BYTE **)(a2 + 32);
  if ( (*a1 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
  {
    if ( (unsigned __int64)(v3 - v4) <= 0xA )
      return sub_CB6200(a2, "LLT_invalid", 0xBu);
    qmemcpy(v4, "LLT_invalid", 11);
    *(_QWORD *)(a2 + 32) += 11LL;
    return 26988;
  }
  if ( (*(_BYTE *)a1 & 4) == 0 )
  {
    if ( (*(_BYTE *)a1 & 6) == 2 )
    {
      if ( v4 == v3 )
      {
        v2 = sub_CB6200(a2, (unsigned __int8 *)"p", 1u);
      }
      else
      {
        *v4 = 112;
        ++*(_QWORD *)(a2 + 32);
      }
      v18 = (*a1 >> 24) & 0xFFFFFF;
      return sub_CB59D0(v2, v18);
    }
    if ( v4 == v3 )
    {
      v2 = sub_CB6200(a2, (unsigned __int8 *)"s", 1u);
      v16 = *a1;
      v17 = *a1 >> 3;
      if ( (*(_BYTE *)a1 & 2) == 0 )
        goto LABEL_22;
    }
    else
    {
      *v4 = 115;
      ++*(_QWORD *)(a2 + 32);
      v16 = *a1;
      v17 = *a1 >> 3;
      if ( (*(_BYTE *)a1 & 2) == 0 )
        goto LABEL_22;
    }
    if ( (v16 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
    {
      v18 = HIWORD(v16);
      return sub_CB59D0(v2, v18);
    }
LABEL_22:
    v18 = v17 >> 29;
    return sub_CB59D0(v2, v18);
  }
  if ( v4 == v3 )
  {
    sub_CB6200(a2, "<", 1u);
  }
  else
  {
    *v4 = 60;
    ++*(_QWORD *)(a2 + 32);
  }
  v6 = (unsigned __int16)((unsigned int)*a1 >> 8);
  if ( (*a1 & 8) != 0 )
  {
    v19 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v19) <= 8 )
    {
      sub_CB6200(a2, "vscale x ", 9u);
    }
    else
    {
      *(_BYTE *)(v19 + 8) = 32;
      *(_QWORD *)v19 = 0x7820656C61637376LL;
      *(_QWORD *)(a2 + 32) += 9LL;
    }
  }
  sub_CB59D0(a2, v6);
  v7 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v7) <= 2 )
  {
    v2 = sub_CB6200(a2, (unsigned __int8 *)" x ", 3u);
  }
  else
  {
    *(_BYTE *)(v7 + 2) = 32;
    *(_WORD *)v7 = 30752;
    *(_QWORD *)(a2 + 32) += 3LL;
  }
  v8 = *a1;
  v9 = *a1 >> 3;
  if ( (*(_BYTE *)a1 & 2) != 0 )
  {
    v10 = HIDWORD(v8);
    v11 = (v8 & 0xFFFFFFFFFFFFFFF9LL) == 0;
    v12 = HIWORD(v8);
    if ( v11 )
      v12 = v10;
    v13 = 0;
    v14 = (v12 << 45) & 0x1FFFE00000000000LL | v9 & 0x1FFFFFE00000LL;
    v15 = 1;
  }
  else
  {
    v14 = v9 & 0xFFFFFFFFE0000000LL;
    v15 = 0;
    v13 = 1;
  }
  v20[0] = (8 * v14) | v13 | (2 * v15);
  sub_34B2640(v20, v2);
  result = *(_QWORD *)(v2 + 32);
  if ( *(_QWORD *)(v2 + 24) == result )
    return sub_CB6200(v2, (unsigned __int8 *)">", 1u);
  *(_BYTE *)result = 62;
  ++*(_QWORD *)(v2 + 32);
  return result;
}
