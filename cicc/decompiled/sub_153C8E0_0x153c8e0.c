// Function: sub_153C8E0
// Address: 0x153c8e0
//
bool __fastcall sub_153C8E0(__int64 **a1)
{
  __int64 v1; // rdx
  char v2; // al

  v1 = **a1;
  v2 = *(_BYTE *)(v1 + 8);
  if ( v2 == 16 )
    v2 = *(_BYTE *)(**(_QWORD **)(v1 + 16) + 8LL);
  return v2 == 11;
}
