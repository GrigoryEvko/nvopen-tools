// Function: sub_2D35090
// Address: 0x2d35090
//
unsigned __int64 __fastcall sub_2D35090(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4)
{
  _DWORD *v6; // rbx
  unsigned __int64 result; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9

  v6 = *(_DWORD **)a1;
  if ( !*(_DWORD *)(*(_QWORD *)a1 + 192LL) )
  {
    result = sub_2D28A50(
               *(_QWORD *)a1,
               (unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4),
               v6[49],
               a2,
               a3,
               a4);
    if ( (unsigned int)result <= 0x10 )
    {
      v6[49] = result;
      *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = result;
      return result;
    }
    v8 = sub_2D2A990(v6, *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4));
    sub_F038C0((unsigned int *)(a1 + 8), (__int64)(v6 + 2), v6[49], v8, v9, v10);
  }
  return sub_2D349B0((unsigned int *)a1, a2, a3, a4);
}
