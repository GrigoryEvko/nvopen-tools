// Function: sub_20FF400
// Address: 0x20ff400
//
void __fastcall sub_20FF400(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  unsigned int v7; // eax
  __int64 v8; // rax

  v6 = *(_QWORD *)a1;
  if ( !*(_DWORD *)(*(_QWORD *)a1 + 192LL) )
  {
    v7 = sub_20FCAC0(
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
    v8 = sub_20FCC80((_QWORD *)v6, *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4));
    sub_3945C20(a1 + 8, v6 + 8, *(unsigned int *)(v6 + 196), v8);
  }
  sub_20FEBD0(a1, a2, a3, a4);
}
