// Function: sub_29F3D50
// Address: 0x29f3d50
//
void __fastcall sub_29F3D50(__int64 a1, __int64 a2)
{
  __int64 v2; // [rsp-20h] [rbp-20h]

  if ( *(_DWORD *)(a1 + 32) == 39 && *(_DWORD *)(a1 + 52) == 3 )
  {
    v2 = sub_BAA610(*(_QWORD *)(a2 + 40));
    if ( BYTE4(v2) )
    {
      if ( (unsigned int)(v2 - 3) <= 1 )
        sub_B30310(a2, 4);
    }
  }
}
