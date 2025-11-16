// Function: sub_252A550
// Address: 0x252a550
//
__int64 __fastcall sub_252A550(__int64 a1, __m128i *a2, __int64 a3, unsigned int a4, bool *a5)
{
  unsigned int v9; // r8d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  char v14; // dl
  bool v15; // dl
  __int64 v16; // rax
  __int64 v17; // rsi
  int v18; // eax
  int v19; // eax
  _DWORD v20[13]; // [rsp+Ch] [rbp-34h] BYREF

  *a5 = 0;
  if ( !(_BYTE)a4 )
  {
    if ( (unsigned int)*(unsigned __int8 *)sub_250D070(a2) - 12 > 1 )
    {
      v20[0] = 51;
      if ( !(unsigned __int8)sub_2516400(a1, a2, (__int64)v20, 1, 1, 51) )
      {
        v11 = sub_25294B0(a1, a2->m128i_i64[0], a2->m128i_i64[1], a3, 1, 0, 1);
        if ( v11 && (*(_BYTE *)(v11 + 97) & 2) != 0 )
        {
          v9 = 1;
          *a5 = (*(_BYTE *)(v11 + 96) & 2) != 0;
          return v9;
        }
        goto LABEL_14;
      }
    }
LABEL_3:
    *a5 = 1;
    return 1;
  }
  if ( (unsigned int)*(unsigned __int8 *)sub_250D070(a2) - 12 <= 1 )
    goto LABEL_3;
  v20[0] = 50;
  if ( (unsigned __int8)sub_2516400(a1, a2, (__int64)v20, 1, 1, 50) )
    goto LABEL_3;
  v12 = sub_25294B0(a1, a2->m128i_i64[0], a2->m128i_i64[1], a3, 1, 0, 1);
  if ( v12 && (*(_BYTE *)(v12 + 97) & 3) == 3 )
  {
    v9 = a4;
    *a5 = (*(_BYTE *)(v12 + 96) & 3) == 3;
    return v9;
  }
LABEL_14:
  if ( (unsigned __int8)(sub_2509800(a2) - 4) <= 1u )
  {
    v16 = sub_252A070(a1, a2->m128i_i64[0], a2->m128i_i64[1], a3, 2, 0, 1);
    v17 = v16;
    if ( v16 )
    {
      v18 = *(_DWORD *)(v16 + 100);
      if ( (unsigned __int8)v18 == 255 || (v18 & 0xFC) == 0xFC )
      {
        v19 = *(unsigned __int8 *)(v17 + 96);
        *a5 = v19 == 255;
        if ( v19 != 255 )
        {
          sub_250ED80(a1, v17, a3, 1);
          return 1;
        }
        return 1;
      }
    }
  }
  v13 = sub_25294B0(a1, a2->m128i_i64[0], a2->m128i_i64[1], a3, 2, 0, 1);
  v9 = 0;
  if ( !v13 )
    return v9;
  if ( (*(_BYTE *)(v13 + 97) & 3) == 3 )
  {
    v14 = *(_BYTE *)(v13 + 96);
    if ( (_BYTE)a4 )
    {
      v15 = (v14 & 3) == 3;
LABEL_21:
      *a5 = v15;
      if ( !v15 )
        sub_250ED80(a1, v13, a3, 1);
      return 1;
    }
LABEL_20:
    v15 = (v14 & 2) != 0;
    goto LABEL_21;
  }
  if ( !(_BYTE)a4 && (*(_BYTE *)(v13 + 97) & 2) != 0 )
  {
    v14 = *(_BYTE *)(v13 + 96);
    goto LABEL_20;
  }
  return v9;
}
