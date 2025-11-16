// Function: sub_749FD0
// Address: 0x749fd0
//
void __fastcall sub_749FD0(__int64 a1, int a2, void (__fastcall **a3)(char *, _QWORD), __int64 a4, __int64 a5)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rcx
  __int64 v11; // r8

  if ( a2 )
  {
    (*a3)("(", a3);
    sub_748000(*(_QWORD *)(a1 + 184), 0, (__int64)a3, v6, v7);
    (*a3)(" - ", a3);
    sub_748000(*(_QWORD *)(a1 + 176), 0, (__int64)a3, v8, v9);
    (*a3)(")", a3);
  }
  else
  {
    sub_748000(*(_QWORD *)(a1 + 184), 0, (__int64)a3, a4, a5);
    (*a3)(" - ", a3);
    sub_748000(*(_QWORD *)(a1 + 176), 0, (__int64)a3, v10, v11);
  }
}
