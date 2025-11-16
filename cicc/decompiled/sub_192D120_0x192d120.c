// Function: sub_192D120
// Address: 0x192d120
//
__int64 __fastcall sub_192D120(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 result; // rax

  result = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, a5, a6);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
