// Function: sub_AC31F0
// Address: 0xac31f0
//
__int64 __fastcall sub_AC31F0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 v2; // dl
  int v4; // eax
  __int64 v5; // [rsp+0h] [rbp-8h]

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_BYTE *)(v1 + 8);
  if ( v2 == 16 )
  {
    BYTE4(v5) = 0;
    LODWORD(v5) = *(_QWORD *)(v1 + 32);
    return v5;
  }
  else
  {
    if ( (unsigned int)v2 - 17 > 1 )
    {
      v4 = *(_DWORD *)(v1 + 12);
      BYTE4(v5) = 0;
    }
    else
    {
      v4 = *(_DWORD *)(v1 + 32);
      BYTE4(v5) = v2 == 18;
    }
    LODWORD(v5) = v4;
    return v5;
  }
}
