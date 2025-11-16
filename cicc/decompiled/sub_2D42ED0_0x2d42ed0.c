// Function: sub_2D42ED0
// Address: 0x2d42ed0
//
__int64 __fastcall sub_2D42ED0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  unsigned __int8 v3; // al

  v2 = 1;
  v3 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
  if ( v3 > 3u && v3 != 5 )
    return (v3 & 0xFD) == 4;
  return v2;
}
