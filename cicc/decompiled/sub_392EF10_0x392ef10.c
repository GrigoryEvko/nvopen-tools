// Function: sub_392EF10
// Address: 0x392ef10
//
void __fastcall sub_392EF10(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rsi
  __int64 v3; // [rsp+8h] [rbp-8h] BYREF

  v3 = a2;
  v2 = *(_BYTE **)(a1 + 96);
  if ( v2 == *(_BYTE **)(a1 + 104) )
  {
    sub_392ED80(a1 + 88, v2, &v3);
  }
  else
  {
    if ( v2 )
    {
      *(_QWORD *)v2 = v3;
      v2 = *(_BYTE **)(a1 + 96);
    }
    *(_QWORD *)(a1 + 96) = v2 + 8;
  }
}
