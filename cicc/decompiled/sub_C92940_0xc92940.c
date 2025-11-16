// Function: sub_C92940
// Address: 0xc92940
//
__int64 __fastcall sub_C92940(__int64 a1, const void *a2, size_t a3)
{
  int v4; // eax
  int v5; // eax
  __int64 *v6; // rax
  __int64 v7; // r8

  v4 = sub_C92610();
  v5 = sub_C92860((__int64 *)a1, a2, a3, v4);
  if ( v5 == -1 )
    return 0;
  v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * v5);
  v7 = *v6;
  *v6 = -8;
  --*(_DWORD *)(a1 + 12);
  ++*(_DWORD *)(a1 + 16);
  return v7;
}
