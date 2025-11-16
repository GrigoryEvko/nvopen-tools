// Function: sub_1181A30
// Address: 0x1181a30
//
__int64 __fastcall sub_1181A30(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( v2 && !*(_QWORD *)(v2 + 8) && (unsigned __int8)(*(_BYTE *)a2 - 42) <= 0x11u )
  {
    v3 = 1;
    **a1 = a2;
  }
  return v3;
}
