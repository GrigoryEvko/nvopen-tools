// Function: sub_131C760
// Address: 0x131c760
//
__int64 __fastcall sub_131C760(__int64 a1)
{
  unsigned int v1; // r12d

  v1 = sub_130AF40(a1);
  if ( !(_BYTE)v1 )
  {
    *(_QWORD *)(a1 + 192) = 0;
    sub_133F510(a1 + 200, "bin");
    *(_QWORD *)(a1 + 216) = 0;
    *(_OWORD *)(a1 + 112) = 0;
    *(_OWORD *)(a1 + 128) = 0;
    *(_OWORD *)(a1 + 144) = 0;
    *(_OWORD *)(a1 + 160) = 0;
    *(_OWORD *)(a1 + 176) = 0;
  }
  return v1;
}
