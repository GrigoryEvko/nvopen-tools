// Function: sub_271DE80
// Address: 0x271de80
//
void __fastcall sub_271DE80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  char v5; // al

  v5 = *(_BYTE *)(a1 + 2);
  if ( v5 == 2 )
  {
    if ( (unsigned __int8)sub_3181380(a2, a3, a4, a5) )
      sub_271D2E0(a1, 3);
  }
  else if ( (unsigned __int8)(v5 - 4) <= 1u )
  {
    BUG();
  }
}
