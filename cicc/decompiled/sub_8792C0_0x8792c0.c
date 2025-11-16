// Function: sub_8792C0
// Address: 0x8792c0
//
__int64 __fastcall sub_8792C0(__int64 a1)
{
  char v1; // al
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 80);
  if ( v1 == 16 )
  {
    a1 = **(_QWORD **)(a1 + 88);
    v1 = *(_BYTE *)(a1 + 80);
  }
  if ( v1 == 24 )
  {
    a1 = *(_QWORD *)(a1 + 88);
    v1 = *(_BYTE *)(a1 + 80);
  }
  if ( (unsigned __int8)(v1 - 10) <= 1u )
  {
    for ( result = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL);
          *(_BYTE *)(result + 140) == 12;
          result = *(_QWORD *)(result + 160) )
    {
      ;
    }
  }
  else
  {
    if ( v1 != 20 )
      sub_721090();
    for ( result = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 176LL) + 152LL);
          *(_BYTE *)(result + 140) == 12;
          result = *(_QWORD *)(result + 160) )
    {
      ;
    }
  }
  return result;
}
