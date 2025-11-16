// Function: sub_39C7390
// Address: 0x39c7390
//
__int64 __fastcall sub_39C7390(__int64 a1)
{
  unsigned __int16 v1; // ax
  __int64 v2; // rdi
  int v3; // ebx

  v1 = sub_398C0A0(*(_QWORD *)(a1 + 200));
  v2 = *(_QWORD *)(a1 + 200);
  v3 = 0;
  if ( v1 > 4u )
    v3 = 8 * (*(_BYTE *)(v2 + 4513) != 0);
  return v3 + (unsigned int)((unsigned __int16)sub_398C0A0(v2) > 4u) + 7;
}
