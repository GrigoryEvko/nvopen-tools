// Function: sub_254E2A0
// Address: 0x254e2a0
//
__int64 __fastcall sub_254E2A0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __m128i v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  int v12; // edx
  int v13; // ecx
  unsigned int v14; // edx
  __int64 v15; // rdi
  int v16; // r8d
  char v17; // [rsp+Fh] [rbp-41h] BYREF
  __m128i v18; // [rsp+10h] [rbp-40h] BYREF
  _BYTE *v19; // [rsp+20h] [rbp-30h]
  __int64 v20; // [rsp+28h] [rbp-28h]

  result = 1;
  if ( *(_BYTE *)(*(_QWORD *)(a3 + 8) + 8LL) == 7 || !*(_QWORD *)(a3 + 16) )
    return result;
  if ( *(_BYTE *)a3 <= 0x15u )
    return sub_252FFB0(
             a2,
             (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2535020,
             (__int64)&v18,
             a1,
             a3,
             0,
             0,
             0,
             0,
             0);
  if ( *(_BYTE *)a3 > 0x1Cu )
  {
    v7 = sub_B43CB0(a3);
    v8 = *(_QWORD *)(a2 + 200);
    if ( *(_DWORD *)(v8 + 40) )
    {
      v11 = *(_QWORD *)(v8 + 8);
      v12 = *(_DWORD *)(v8 + 24);
      if ( v12 )
      {
        v13 = v12 - 1;
        v14 = (v12 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v15 = *(_QWORD *)(v11 + 8LL * v14);
        if ( v7 == v15 )
          goto LABEL_6;
        v16 = 1;
        while ( v15 != -4096 )
        {
          v14 = v13 & (v16 + v14);
          v15 = *(_QWORD *)(v11 + 8LL * v14);
          if ( v7 == v15 )
            goto LABEL_6;
          ++v16;
        }
      }
      return 0;
    }
  }
LABEL_6:
  v17 = 0;
  v9.m128i_i64[0] = sub_250D2C0(a3, 0);
  v18 = v9;
  v19 = sub_2527570(a2, &v18, a1, &v17);
  v20 = v10;
  result = (unsigned __int8)v10;
  if ( !(_BYTE)v10 )
    return 1;
  if ( !v19 )
    return sub_252FFB0(
             a2,
             (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2535020,
             (__int64)&v18,
             a1,
             a3,
             0,
             0,
             0,
             0,
             0);
  return result;
}
