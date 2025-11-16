// Function: sub_2E1D0D0
// Address: 0x2e1d0d0
//
void __fastcall sub_2E1D0D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  unsigned int v7; // eax
  unsigned __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9

  v6 = *(_QWORD *)a1;
  if ( !*(_DWORD *)(*(_QWORD *)a1 + 192LL) )
  {
    v7 = sub_2E1A1C0(
           *(_QWORD *)a1,
           (unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4),
           *(_DWORD *)(v6 + 196),
           a2,
           a3,
           a4);
    if ( v7 <= 8 )
    {
      *(_DWORD *)(v6 + 196) = v7;
      *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v7;
      return;
    }
    v8 = sub_2E1A380((_QWORD *)v6, *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4));
    sub_F038C0((unsigned int *)(a1 + 8), v6 + 8, *(_DWORD *)(v6 + 196), v8, v9, v10);
  }
  sub_2E1C970((unsigned int *)a1, a2, a3, a4);
}
