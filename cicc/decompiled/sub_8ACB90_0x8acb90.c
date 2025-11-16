// Function: sub_8ACB90
// Address: 0x8acb90
//
__int64 __fastcall sub_8ACB90(__int64 a1, char a2, __int64 *a3, int a4, int a5, int a6, int a7)
{
  __int64 result; // rax
  __int64 v11; // r14
  int v12; // eax
  char v13; // dl
  char v14; // dl
  __int64 v15; // rsi
  int v16; // edx
  char v17; // r13
  int v18; // r13d
  char v19; // al
  __int64 v20; // rsi
  char v21; // al
  int v22; // edi
  char v23; // dl
  __int64 v24; // rsi
  char v25; // dl
  _BYTE *v26; // rdi
  int v27; // edx
  unsigned int v28; // eax
  char v29; // dl
  __int64 v30; // rdx
  char v31; // al
  _BYTE *v32; // rdi
  __int64 v33; // rdx
  char v34; // al
  int v36; // [rsp+8h] [rbp-38h]

  result = sub_892240(a1);
  if ( !result )
    return result;
  v11 = result;
  v12 = sub_8919F0(result, 0);
  if ( a6 | a5 || (v13 = *(_BYTE *)(v11 + 80), (v13 & 8) == 0) )
  {
    if ( a2 == 15 )
    {
      v22 = *(_DWORD *)(v11 + 92);
      v23 = *(_BYTE *)(v11 + 80) & 0xE7 | (16 * a4) & 0x10 | 8;
      *(_BYTE *)(v11 + 80) = v23;
      v24 = *a3;
      *(_BYTE *)(v11 + 80) = v23 & 0xDF;
      *(_QWORD *)(v11 + 84) = v24;
      if ( v22 )
        goto LABEL_24;
      goto LABEL_23;
    }
    if ( a2 != 16 )
    {
LABEL_13:
      v17 = *(_BYTE *)(v11 + 80);
      *(_BYTE *)(v11 + 80) = v17 | 0x40;
      v18 = v17 & 1;
      goto LABEL_14;
    }
    if ( !v12 || a6 )
    {
      v18 = 0;
      *(_BYTE *)(v11 + 80) = *(_BYTE *)(v11 + 80) & 0xC7 | 0x20;
    }
    else
    {
      v18 = *(_BYTE *)(v11 + 80) & 1;
    }
    v29 = *(_BYTE *)(a1 + 80);
    if ( ((v29 - 7) & 0xFD) != 0 )
    {
      if ( a5 )
      {
        *(_BYTE *)(*(_QWORD *)(a1 + 88) + 195LL) |= 6u;
        goto LABEL_16;
      }
    }
    else
    {
      if ( a5 )
      {
        *(_WORD *)(*(_QWORD *)(a1 + 88) + 170LL) |= 0x180u;
        goto LABEL_16;
      }
      if ( v29 == 7 || v29 == 9 )
      {
        v30 = *(_QWORD *)(a1 + 88);
        v31 = *(_BYTE *)(v30 + 172);
        if ( (v31 & 0x20) != 0 )
          *(_BYTE *)(v30 + 172) = v31 | 0x80;
        goto LABEL_14;
      }
    }
    if ( !v12 )
      goto LABEL_14;
    v32 = *(_BYTE **)(a1 + 88);
    if ( (char)v32[192] >= 0 )
    {
      if ( (v32[195] & 1) == 0 )
        goto LABEL_45;
      if ( !(unsigned int)sub_736960((__int64)v32) )
      {
        v32 = *(_BYTE **)(a1 + 88);
        goto LABEL_45;
      }
    }
    if ( !dword_4D04824 )
      goto LABEL_14;
    v32 = *(_BYTE **)(a1 + 88);
    if ( v32[172] == 2 )
      goto LABEL_14;
LABEL_45:
    v32[203] |= 0x40u;
    *(_BYTE *)(*(_QWORD *)(a1 + 88) + 204LL) &= ~1u;
    goto LABEL_14;
  }
  if ( a2 != 16 )
  {
    if ( a7 || ((v13 & 0x10) != 0) != a4 )
    {
      v36 = v12;
      sub_685440(7u, 0x2F8u, a1);
      v12 = v36;
    }
    if ( a2 == 15 )
    {
      v14 = *(_BYTE *)(v11 + 80) & 0xE7 | (16 * a4) & 0x10 | 8;
      *(_BYTE *)(v11 + 80) = v14;
      v15 = *a3;
      *(_BYTE *)(v11 + 80) = v14 & 0xDF;
      v16 = *(_DWORD *)(v11 + 92);
      *(_QWORD *)(v11 + 84) = v15;
      if ( v16 )
      {
LABEL_25:
        v25 = *(_BYTE *)(a1 + 80);
        if ( v25 == 7 || v25 == 9 )
        {
          v33 = *(_QWORD *)(a1 + 88);
          v34 = *(_BYTE *)(v33 + 172);
          if ( (v34 & 0x20) != 0 )
          {
            v18 = 1;
            *(_BYTE *)(v33 + 172) = v34 & 0x7F;
            goto LABEL_14;
          }
        }
        else if ( v12 )
        {
          v26 = *(_BYTE **)(a1 + 88);
          if ( (char)v26[192] >= 0 )
          {
            if ( (v26[195] & 1) == 0 )
            {
LABEL_30:
              v26[203] &= ~0x40u;
              v18 = 1;
              *(_BYTE *)(*(_QWORD *)(a1 + 88) + 204LL) |= 1u;
              sub_7604D0(*(_QWORD *)(a1 + 88), 0xBu);
              goto LABEL_14;
            }
            if ( !(unsigned int)sub_736960((__int64)v26) )
            {
              v26 = *(_BYTE **)(a1 + 88);
              goto LABEL_30;
            }
          }
          if ( dword_4D04824 )
          {
            v26 = *(_BYTE **)(a1 + 88);
            if ( v26[172] != 2 )
              goto LABEL_30;
          }
        }
        v18 = 1;
LABEL_14:
        if ( !(a5 | a4) )
          sub_88F6E0(a1);
        goto LABEL_16;
      }
LABEL_23:
      *(_QWORD *)(v11 + 92) = *a3;
LABEL_24:
      v18 = 1;
      if ( !a5 )
        goto LABEL_25;
LABEL_16:
      sub_8AC530(v11, v18, 0);
      v19 = *(_BYTE *)(a1 + 80);
      v20 = *(_QWORD *)(a1 + 88);
      if ( v19 == 7 || v19 == 9 )
      {
        v27 = (*(_BYTE *)(v11 + 80) >> 2) & 2 | *(_BYTE *)(v20 + 171) & 0xFD;
        *(_BYTE *)(v20 + 171) = v27;
        v28 = v27 & 0xFFFFFFFB | (*(_BYTE *)(v11 + 80) >> 2) & 4;
        *(_BYTE *)(v20 + 171) = v28;
        result = (*(_BYTE *)(v11 + 80) >> 2) & 8 | v28 & 0xFFFFFFF7;
        *(_BYTE *)(v20 + 171) = result;
      }
      else
      {
        v21 = (8 * *(_BYTE *)(v11 + 80)) & 0x40 | *(_BYTE *)(v20 + 195) & 0xBF;
        *(_BYTE *)(v20 + 195) = v21;
        *(_BYTE *)(v20 + 195) = (*(_BYTE *)(v11 + 80) >> 4 << 7) | v21 & 0x7F;
        *(_BYTE *)(v20 + 196) = ((*(_BYTE *)(v11 + 80) & 0x20) != 0) | *(_BYTE *)(v20 + 196) & 0xFE;
        result = (unsigned int)*(unsigned __int8 *)(v20 + 174) - 1;
        if ( (unsigned __int8)(*(_BYTE *)(v20 + 174) - 1) <= 1u )
        {
          for ( result = *(_QWORD *)(v20 + 176); result; result = *(_QWORD *)result )
          {
            *(_BYTE *)(*(_QWORD *)(result + 8) + 195LL) = (8 * *(_BYTE *)(v11 + 80)) & 0x40
                                                        | *(_BYTE *)(*(_QWORD *)(result + 8) + 195LL) & 0xBF;
            *(_BYTE *)(*(_QWORD *)(result + 8) + 195LL) = (*(_BYTE *)(v11 + 80) >> 4 << 7)
                                                        | *(_BYTE *)(*(_QWORD *)(result + 8) + 195LL) & 0x7F;
            *(_BYTE *)(*(_QWORD *)(result + 8) + 196LL) = ((*(_BYTE *)(v11 + 80) & 0x20) != 0)
                                                        | *(_BYTE *)(*(_QWORD *)(result + 8) + 196LL) & 0xFE;
          }
        }
      }
      return result;
    }
    goto LABEL_13;
  }
  result = sub_685440(dword_4F077BC == 0 ? 7 : 5, 0x641u, a1);
  if ( !a4 )
    return sub_88F6E0(a1);
  return result;
}
