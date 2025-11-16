// Function: sub_5E4B20
// Address: 0x5e4b20
//
__int64 __fastcall sub_5E4B20(__int64 a1)
{
  __int64 v1; // r12

  v1 = qword_4CF8000;
  if ( qword_4CF8000 )
    qword_4CF8000 = *(_QWORD *)qword_4CF8000;
  else
    v1 = sub_823970(192);
  *(_BYTE *)(v1 + 184) &= 0xC0u;
  *(_QWORD *)(v1 + 8) = a1;
  *(_QWORD *)v1 = 0;
  *(_QWORD *)(v1 + 16) = 0;
  *(_QWORD *)(v1 + 128) = 0;
  *(_QWORD *)(v1 + 136) = 0;
  *(_QWORD *)(v1 + 144) = 0;
  sub_87E3B0(v1 + 24);
  sub_7ADF70(v1 + 152, 1);
  return v1;
}
