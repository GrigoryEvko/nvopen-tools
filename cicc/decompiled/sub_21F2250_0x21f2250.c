// Function: sub_21F2250
// Address: 0x21f2250
//
__int64 __fastcall sub_21F2250(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax

  v6 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v6 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, a5, a6);
    v6 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v6) = &unk_4FC6A0E;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4F9A488);
  return sub_1636A10(a2, (__int64)&unk_4F9A488);
}
