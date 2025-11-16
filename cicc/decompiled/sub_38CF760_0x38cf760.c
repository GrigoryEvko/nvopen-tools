// Function: sub_38CF760
// Address: 0x38cf760
//
__int64 __fastcall sub_38CF760(__int64 a1, char a2, char a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = a2;
  *(_BYTE *)(a1 + 17) = a3;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = -1;
  if ( a2 != 13 )
  {
    if ( a4 )
    {
      v4 = *(_QWORD *)(a4 + 96);
      *(_QWORD *)(a1 + 8) = a4 + 96;
      v4 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)a1 = v4;
      *(_QWORD *)(v4 + 8) = a1;
      result = *(_QWORD *)(a4 + 96) & 7LL;
      *(_QWORD *)(a4 + 96) = result | a1;
    }
  }
  return result;
}
