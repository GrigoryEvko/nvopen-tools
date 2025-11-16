// Function: sub_C54F80
// Address: 0xc54f80
//
__int64 __fastcall sub_C54F80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, _BYTE *a7)
{
  __int64 v8; // rbx
  __int64 v9; // rcx
  __m128i v12[3]; // [rsp+10h] [rbp-B0h] BYREF
  __m128i v13[2]; // [rsp+40h] [rbp-80h] BYREF
  char v14; // [rsp+60h] [rbp-60h]
  char v15; // [rsp+61h] [rbp-5Fh]
  __m128i v16; // [rsp+70h] [rbp-50h] BYREF
  __int64 v17; // [rsp+80h] [rbp-40h]
  __int64 v18; // [rsp+88h] [rbp-38h]
  __int16 v19; // [rsp+90h] [rbp-30h]

  if ( !a6 )
    goto LABEL_11;
  if ( a6 == 4 )
  {
    if ( *(_DWORD *)a5 != 1702195828 && *(_DWORD *)a5 != 1163219540 && *(_DWORD *)a5 != 1702195796 )
      goto LABEL_17;
    goto LABEL_11;
  }
  if ( a6 != 1 )
  {
    if ( a6 == 5
      && (*(_DWORD *)a5 == 1936482662 && *(_BYTE *)(a5 + 4) == 101
       || *(_DWORD *)a5 == 1397506374 && *(_BYTE *)(a5 + 4) == 69
       || *(_DWORD *)a5 == 1936482630 && *(_BYTE *)(a5 + 4) == 101) )
    {
      goto LABEL_14;
    }
    goto LABEL_17;
  }
  if ( *(_BYTE *)a5 == 49 )
  {
LABEL_11:
    *a7 = 1;
    return 0;
  }
  if ( *(_BYTE *)a5 == 48 )
  {
LABEL_14:
    *a7 = 0;
    return 0;
  }
LABEL_17:
  v8 = sub_CEADF0(a1, a2, a3, a4, a5, a6);
  v15 = 1;
  v13[0].m128i_i64[0] = (__int64)"' is invalid value for boolean argument! Try 0 or 1";
  v19 = 1283;
  v17 = a5;
  v14 = 3;
  v16.m128i_i64[0] = (__int64)"'";
  v18 = a6;
  sub_9C6370(v12, &v16, v13, v9, a5, a6);
  return sub_C53280(a2, (__int64)v12, 0, 0, v8);
}
