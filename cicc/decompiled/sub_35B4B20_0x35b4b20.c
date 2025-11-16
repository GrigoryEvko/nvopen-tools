// Function: sub_35B4B20
// Address: 0x35b4b20
//
__int64 __fastcall sub_35B4B20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rdi
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdi

  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 16);
  v5 = *(_QWORD **)a2;
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 16) = v5;
  *(_QWORD *)(a1 + 32) = a3;
  *(_QWORD *)(a1 + 40) = a4;
  sub_2EBF200(v5);
  sub_2F5FFA0((_QWORD *)(a1 + 48), *(_QWORD *)(a2 + 24));
  v6 = *(_QWORD *)(a1 + 728);
  *(_DWORD *)(a1 + 696) = 0;
  while ( v6 )
  {
    sub_35B4950(*(_QWORD *)(v6 + 24));
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
    j_j___libc_free_0(v7);
  }
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 736) = a1 + 720;
  *(_QWORD *)(a1 + 744) = a1 + 720;
  *(_QWORD *)(a1 + 752) = 0;
  return a1 + 720;
}
