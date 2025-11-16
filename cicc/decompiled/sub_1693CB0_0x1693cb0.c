// Function: sub_1693CB0
// Address: 0x1693cb0
//
__int64 *__fastcall sub_1693CB0(__int64 *a1, int a2)
{
  __int64 v2; // rax

  v2 = sub_22077B0(16);
  if ( v2 )
  {
    *(_DWORD *)(v2 + 8) = a2;
    *(_QWORD *)v2 = &unk_49EEA60;
  }
  *a1 = v2 | 1;
  return a1;
}
