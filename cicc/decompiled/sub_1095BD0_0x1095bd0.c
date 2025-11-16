// Function: sub_1095BD0
// Address: 0x1095bd0
//
void __fastcall sub_1095BD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  *(_QWORD *)(a1 + 160) = a2;
  if ( !a4 )
    a4 = a2;
  *(_QWORD *)(a1 + 168) = a3;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 152) = a4;
  *(_BYTE *)(a1 + 179) = a5;
}
