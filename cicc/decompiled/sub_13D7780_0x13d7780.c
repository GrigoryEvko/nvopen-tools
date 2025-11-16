// Function: sub_13D7780
// Address: 0x13d7780
//
__int64 __fastcall sub_13D7780(_QWORD **a1, _BYTE *a2)
{
  __int64 v3; // rax

  if ( a2[16] == 13 )
  {
    **a1 = a2 + 24;
    return 1;
  }
  else if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 && (v3 = sub_15A1020(a2)) != 0 && *(_BYTE *)(v3 + 16) == 13 )
  {
    **a1 = v3 + 24;
    return 1;
  }
  else
  {
    return 0;
  }
}
