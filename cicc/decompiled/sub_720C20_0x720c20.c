// Function: sub_720C20
// Address: 0x720c20
//
__int64 __fastcall sub_720C20(_QWORD *a1, const char *a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  char v5; // si
  char v6; // dl
  size_t v7; // rax
  __int64 result; // rax

  sub_823950();
  v3 = a1[2];
  if ( v3 )
  {
    v4 = a1[4];
    v5 = *(_BYTE *)(v4 + v3 - 1);
    v6 = v5 != 47;
    if ( unk_4F07598 )
      v6 &= v5 != 92;
    if ( v6 )
    {
      if ( (unsigned __int64)(v3 + 1) > a1[1] )
      {
        sub_823810(a1);
        v4 = a1[4];
        v3 = a1[2];
      }
      *(_BYTE *)(v4 + v3) = 47;
      ++a1[2];
    }
  }
  v7 = strlen(a2);
  sub_8238B0(a1, a2, v7);
  result = a1[2];
  if ( (unsigned __int64)(result + 1) > a1[1] )
  {
    sub_823810(a1);
    result = a1[2];
  }
  *(_BYTE *)(a1[4] + result) = 0;
  ++a1[2];
  return result;
}
