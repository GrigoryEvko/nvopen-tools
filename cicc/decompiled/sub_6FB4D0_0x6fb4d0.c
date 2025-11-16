// Function: sub_6FB4D0
// Address: 0x6fb4d0
//
__int64 __fastcall sub_6FB4D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdi
  char v5; // dl
  __int64 v6; // rax

  sub_6FB450(a1, a2);
  result = 0;
  if ( *(_BYTE *)(a1 + 16) )
  {
    v4 = *(_QWORD *)a1;
    v5 = *(_BYTE *)(*(_QWORD *)a1 + 140LL);
    if ( v5 == 12 )
    {
      v6 = *(_QWORD *)a1;
      do
      {
        v6 = *(_QWORD *)(v6 + 160);
        v5 = *(_BYTE *)(v6 + 140);
      }
      while ( v5 == 12 );
    }
    result = 0;
    if ( v5 )
    {
      if ( (unsigned int)sub_8D2E30(v4) )
      {
        return 1;
      }
      else
      {
        sub_6E6930(a2, a1, *(_QWORD *)a1);
        return 0;
      }
    }
  }
  return result;
}
