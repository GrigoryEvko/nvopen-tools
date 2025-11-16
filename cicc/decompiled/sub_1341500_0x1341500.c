// Function: sub_1341500
// Address: 0x1341500
//
unsigned __int64 __fastcall sub_1341500(__int64 *a1, __int64 *a2, unsigned __int64 *a3, __int64 a4, unsigned __int8 a5)
{
  unsigned __int64 result; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // r9
  __int64 v8; // r10
  __int64 v9; // rdx

  result = (unsigned __int64)a3;
  if ( a3 )
  {
    v6 = *a3;
    v7 = (v6 >> 15) & 0x1C;
    v8 = (v6 >> 43) & 2;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  v9 = v7 | v8 | a5 | (unsigned __int64)(a4 << 48) | result & 0xFFFFFFFFFFFFLL;
  *a1 = v9;
  if ( a2 )
    *a2 = v9;
  return result;
}
