// Function: sub_CE9180
// Address: 0xce9180
//
__int64 __fastcall sub_CE9180(__int64 a1)
{
  int v1; // ebx
  int v3; // eax
  __int64 v4; // [rsp+8h] [rbp-18h] BYREF

  if ( (unsigned __int8)sub_CE7ED0(a1, "maxnreg", 7u, &v4) )
  {
    BYTE4(v4) = 1;
    return v4;
  }
  else
  {
    if ( (unsigned __int8)sub_B2D620(a1, "nvvm.maxnreg", 0xCu) )
    {
      v3 = sub_B2D810(a1, "nvvm.maxnreg", 0xCu, 0);
      BYTE4(v4) = 1;
      LODWORD(v4) = v3;
    }
    else
    {
      LODWORD(v4) = v1;
      BYTE4(v4) = 0;
    }
    return v4;
  }
}
