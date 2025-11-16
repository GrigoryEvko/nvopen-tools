// Function: sub_31C5B60
// Address: 0x31c5b60
//
__int64 __fastcall sub_31C5B60(unsigned int *a1, __int64 a2, _DWORD *a3)
{
  unsigned int v4; // r8d
  unsigned int v6; // eax
  char v7; // dl
  unsigned int v8; // r14d
  unsigned int v9; // r8d
  unsigned int v10; // r10d
  unsigned int v11; // r9d
  unsigned int v12; // esi
  __int64 v13; // [rsp+0h] [rbp-30h]

  if ( a3 )
    *a3 = a1[7];
  if ( !a1[3] )
    return a1[1];
  v6 = sub_CE8F50(a2);
  if ( v7 )
  {
    v8 = v6;
    v13 = sub_CE90E0(a2);
    if ( BYTE4(v13) )
    {
      v9 = a1[9];
      if ( (unsigned int)v13 >= v9 )
        v9 = v13;
    }
    else
    {
      v9 = a1[9];
    }
    v10 = a1[8];
    v11 = a1[4];
    v12 = a1[5];
    if ( v10 <= v9 )
      v9 = a1[8];
    v4 = v12 * (v11 * (a1[3] / v9 / v11) / v8 / v12);
    if ( a3 )
      *a3 = v12 * (v11 * (a1[3] / v10 / v11) / v8 / v12);
    return v4;
  }
  v4 = a1[1];
  if ( v4 )
    return v4;
  return a1[6];
}
