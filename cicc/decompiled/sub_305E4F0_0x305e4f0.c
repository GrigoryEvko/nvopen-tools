// Function: sub_305E4F0
// Address: 0x305e4f0
//
__int64 __fastcall sub_305E4F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // rax

  result = 0;
  if ( *(_QWORD *)(a2 + 8) == 8 && **(_QWORD **)a2 == 0x61612D787470766ELL )
  {
    v8 = *(unsigned int *)(a3 + 8);
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v8 + 1, 8u, a5, a6);
      v8 = *(unsigned int *)(a3 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = sub_30608C0;
    ++*(_DWORD *)(a3 + 8);
    return 1;
  }
  return result;
}
