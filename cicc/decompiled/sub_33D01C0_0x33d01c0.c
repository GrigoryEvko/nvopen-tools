// Function: sub_33D01C0
// Address: 0x33d01c0
//
_QWORD *__fastcall sub_33D01C0(__int64 a1, int a2, unsigned __int64 a3, int a4, unsigned __int64 *a5, __int64 a6)
{
  __int64 v6; // r10
  int v7; // eax

  v6 = *(_QWORD *)(a1 + 1024);
  v7 = 0;
  if ( v6 )
    v7 = *(_DWORD *)(v6 + 8);
  return sub_33D00B0(a1, a2, a3, a4, a5, a6, v7);
}
