// Function: sub_155F820
// Address: 0x155f820
//
__int64 __fastcall sub_155F820(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // rsi

  v3 = *a2;
  if ( v3 )
  {
    sub_155F6F0((_QWORD *)a1, v3, a3);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
  }
  return a1;
}
