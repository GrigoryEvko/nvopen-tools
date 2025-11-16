// Function: sub_3007070
// Address: 0x3007070
//
bool __fastcall sub_3007070(__int64 a1)
{
  __int64 v1; // rcx
  int v2; // edx

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(unsigned __int8 *)(v1 + 8);
  if ( (unsigned int)(v2 - 17) <= 1 )
    LOBYTE(v2) = *(_BYTE *)(**(_QWORD **)(v1 + 16) + 8LL);
  return (_BYTE)v2 == 12;
}
