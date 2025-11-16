// Function: sub_3112AB0
// Address: 0x3112ab0
//
__int64 __fastcall sub_3112AB0(__int64 a1)
{
  _QWORD *v1; // rdx
  unsigned int v2; // r8d

  v1 = *(_QWORD **)(a1 + 8);
  v2 = 0;
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v1 > 7u )
    LOBYTE(v2) = *v1 == 0x81617461646763FFLL;
  return v2;
}
