// Function: sub_6E9610
// Address: 0x6e9610
//
__int64 __fastcall sub_6E9610(_BYTE *a1)
{
  __int64 result; // rax
  __int64 v2; // r13
  char v3; // dl
  __int64 v4; // rax
  __int64 v5; // rax

  result = 0;
  if ( a1[16] )
  {
    v2 = *(_QWORD *)a1;
    v3 = *(_BYTE *)(*(_QWORD *)a1 + 140LL);
    if ( v3 == 12 )
    {
      v4 = *(_QWORD *)a1;
      do
      {
        v4 = *(_QWORD *)(v4 + 160);
        v3 = *(_BYTE *)(v4 + 140);
      }
      while ( v3 == 12 );
    }
    result = 0;
    if ( v3 )
    {
      if ( (unsigned int)sub_8D2E30(*(_QWORD *)a1) && (v5 = sub_8D46C0(v2), (unsigned int)sub_8D2310(v5)) )
      {
        return 1;
      }
      else
      {
        sub_6E68E0(0x6Du, (__int64)a1);
        return 0;
      }
    }
  }
  return result;
}
