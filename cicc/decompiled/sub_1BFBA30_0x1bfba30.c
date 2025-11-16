// Function: sub_1BFBA30
// Address: 0x1bfba30
//
__int64 __fastcall sub_1BFBA30(unsigned int *a1, __int64 a2, _DWORD *a3)
{
  unsigned int v4; // r8d
  unsigned int v6; // r8d
  unsigned int v7; // r10d
  unsigned int v8; // r9d
  unsigned int v9; // esi
  unsigned int v10; // [rsp+4h] [rbp-2Ch] BYREF
  unsigned int v11; // [rsp+8h] [rbp-28h] BYREF
  char v12; // [rsp+Ch] [rbp-24h]

  if ( a3 )
    *a3 = a1[7];
  if ( !a1[3] )
    return a1[1];
  sub_1C2ECF0(&v11, a2);
  if ( v12 )
  {
    if ( (unsigned __int8)sub_1C2EF70(a2, &v10) )
    {
      v6 = v10;
      if ( a1[9] >= v10 )
        v6 = a1[9];
    }
    else
    {
      v6 = a1[9];
    }
    v7 = a1[8];
    v8 = a1[4];
    v9 = a1[5];
    if ( v7 <= v6 )
      v6 = a1[8];
    v4 = v9 * (v8 * (a1[3] / v6 / v8) / v11 / v9);
    if ( a3 )
      *a3 = v9 * (v8 * (a1[3] / v7 / v8) / v11 / v9);
    return v4;
  }
  v4 = a1[1];
  if ( v4 )
    return v4;
  return a1[6];
}
