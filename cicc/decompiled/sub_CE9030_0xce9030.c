// Function: sub_CE9030
// Address: 0xce9030
//
__int64 __fastcall sub_CE9030(__int64 a1)
{
  int v1; // ebx
  char v3; // al
  char v4; // dl
  int v5; // eax
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_CE7ED0(a1, "cluster_max_blocks", 0x12u, v6)
    || (unsigned __int8)sub_CE7ED0(a1, "nvvm.maxclusterrank", 0x13u, v6) )
  {
    BYTE4(v6[0]) = 1;
    return v6[0];
  }
  else
  {
    v3 = sub_B2D620(a1, "nvvm.maxclusterrank", 0x13u);
    v4 = 0;
    if ( v3 )
    {
      v5 = sub_B2D810(a1, "nvvm.maxclusterrank", 0x13u, 0);
      v4 = 1;
      v1 = v5;
    }
    LODWORD(v6[0]) = v1;
    BYTE4(v6[0]) = v4;
    return v6[0];
  }
}
