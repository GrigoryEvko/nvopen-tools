// Function: sub_1070390
// Address: 0x1070390
//
__int64 __fastcall sub_1070390(__int64 a1, char a2)
{
  __int64 v2; // rbp
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned int v5; // r8d
  int v6; // ecx
  unsigned int v7; // eax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int8 v21; // [rsp-1D9h] [rbp-1D9h]
  __int64 v22; // [rsp-1D0h] [rbp-1D0h] BYREF
  __m128i v23; // [rsp-1C8h] [rbp-1C8h] BYREF
  __int16 v24; // [rsp-1A8h] [rbp-1A8h]
  __m128i v25; // [rsp-198h] [rbp-198h] BYREF
  char v26; // [rsp-178h] [rbp-178h]
  char v27; // [rsp-177h] [rbp-177h]
  __m128i v28[3]; // [rsp-168h] [rbp-168h] BYREF
  __m128i v29; // [rsp-138h] [rbp-138h] BYREF
  char v30; // [rsp-118h] [rbp-118h]
  char v31; // [rsp-117h] [rbp-117h]
  __m128i v32[3]; // [rsp-108h] [rbp-108h] BYREF
  __m128i v33; // [rsp-D8h] [rbp-D8h] BYREF
  __int16 v34; // [rsp-B8h] [rbp-B8h]
  __m128i v35[3]; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v36; // [rsp-78h] [rbp-78h] BYREF
  char v37; // [rsp-58h] [rbp-58h]
  char v38; // [rsp-57h] [rbp-57h]
  __m128i v39[3]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v40; // [rsp-18h] [rbp-18h]
  __int64 v41; // [rsp-10h] [rbp-10h]
  __int64 v42; // [rsp-8h] [rbp-8h]

  v5 = *(unsigned __int16 *)(a1 + 12);
  if ( (((*(_BYTE *)(a1 + 9) & 0x70) - 48) & 0xE0) == 0 && ((*(_DWORD *)(a1 + 8) >> 15) & 0x1F) != 0 )
  {
    v6 = ((*(_DWORD *)(a1 + 8) >> 15) & 0x1F) - 1;
    if ( (unsigned __int8)v6 > 0xFu )
    {
      v42 = v2;
      v41 = v4;
      v40 = v3;
      v21 = v6;
      v38 = 1;
      v36.m128i_i64[0] = (__int64)"'";
      v37 = 3;
      v33.m128i_i64[0] = sub_E5B9B0(a1);
      v29.m128i_i64[0] = (__int64)"' for '";
      v33.m128i_i64[1] = v9;
      v22 = 1LL << v21;
      v23.m128i_i64[0] = (__int64)&v22;
      v25.m128i_i64[0] = (__int64)"invalid 'common' alignment '";
      v34 = 261;
      v31 = 1;
      v30 = 3;
      v24 = 267;
      v27 = 1;
      v26 = 3;
      sub_9C6370(v28, &v25, &v23, v21, v10, v11);
      sub_9C6370(v32, v28, &v29, v12, v13, v14);
      sub_9C6370(v35, v32, &v33, v15, v16, v17);
      sub_9C6370(v39, v35, &v36, v18, v19, v20);
      sub_C64D30((__int64)v39, 0);
    }
    LOWORD(v5) = v5 & 0xF0FF;
    v5 |= v6 << 8;
  }
  v7 = v5;
  if ( a2 )
  {
    BYTE1(v7) = BYTE1(v5) | 2;
    return v7;
  }
  return v5;
}
