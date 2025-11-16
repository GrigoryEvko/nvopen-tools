// Function: sub_7E2B60
// Address: 0x7e2b60
//
void __fastcall sub_7E2B60(__int64 a1)
{
  char v1; // al

  while ( a1 )
  {
    v1 = *(_BYTE *)(a1 + 40);
    if ( v1 == 17 )
    {
      *(_BYTE *)(*(_QWORD *)(a1 + 72) + 49LL) |= 2u;
    }
    else if ( v1 != 20 )
    {
      return;
    }
    a1 = *(_QWORD *)(a1 + 16);
  }
}
