// Function: sub_39A3560
// Address: 0x39a3560
//
__int64 __fastcall sub_39A3560(__int64 a1, __int64 *a2, __int16 a3, __int64 a4, __int64 a5)
{
  __int16 v5; // ax
  __int64 v7[2]; // [rsp+0h] [rbp-10h] BYREF

  if ( *(_BYTE *)(a4 + 2) )
  {
    v5 = *(_WORD *)a4;
  }
  else
  {
    v5 = 11;
    if ( (a5 & 0xFFFFFFFFFFFFFF00LL) != 0 )
    {
      v5 = 5;
      if ( (a5 & 0xFFFFFFFFFFFF0000LL) != 0 )
        v5 = (a5 != (unsigned int)a5) + 6;
    }
    *(_WORD *)a4 = v5;
    *(_BYTE *)(a4 + 2) = 1;
  }
  WORD2(v7[0]) = a3;
  LODWORD(v7[0]) = 1;
  HIWORD(v7[0]) = v5;
  v7[1] = a5;
  return sub_39A31C0(a2, (__int64 *)(a1 + 88), v7);
}
