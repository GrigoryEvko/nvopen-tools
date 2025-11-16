// Function: sub_1F58CD0
// Address: 0x1f58cd0
//
bool __fastcall sub_1F58CD0(__int64 a1)
{
  __int64 v1; // rdx
  char v2; // al

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_BYTE *)(v1 + 8);
  if ( v2 == 16 )
    v2 = *(_BYTE *)(**(_QWORD **)(v1 + 16) + 8LL);
  return (unsigned __int8)(v2 - 1) <= 5u;
}
