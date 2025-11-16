// Function: sub_2EE7340
// Address: 0x2ee7340
//
__int64 __fastcall sub_2EE7340(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rcx

  v4 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
  *(_DWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = sub_2EE6DD0;
  *(_QWORD *)a1 = v4;
  *(_QWORD *)(a1 + 24) = sub_2EE6E60;
  return a1;
}
