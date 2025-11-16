// Function: sub_3939040
// Address: 0x3939040
//
__int64 *__fastcall sub_3939040(__int64 *a1, int *a2)
{
  int v2; // ebx
  __int64 v3; // rax

  v2 = *a2;
  v3 = sub_22077B0(0x10u);
  if ( v3 )
  {
    *(_DWORD *)(v3 + 8) = v2;
    *(_QWORD *)v3 = &unk_49EEA60;
  }
  *a1 = v3 | 1;
  return a1;
}
