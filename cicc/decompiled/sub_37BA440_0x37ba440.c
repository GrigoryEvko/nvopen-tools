// Function: sub_37BA440
// Address: 0x37ba440
//
__int64 __fastcall sub_37BA440(__int64 a1, unsigned int a2)
{
  unsigned int *v2; // rbx

  v2 = (unsigned int *)(*(_QWORD *)(a1 + 64) + 4LL * a2);
  if ( *v2 == -1 )
    *v2 = sub_37BA230(a1, a2);
  return *v2;
}
