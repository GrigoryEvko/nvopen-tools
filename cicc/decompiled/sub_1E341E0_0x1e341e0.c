// Function: sub_1E341E0
// Address: 0x1e341e0
//
__int64 __fastcall sub_1E341E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rax
  int v6; // edx

  v5 = sub_1EB39F0(*(_QWORD *)(a2 + 376), a3);
  *(_QWORD *)(a1 + 8) = a4;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)a1 = v5 | 4;
  v6 = 0;
  if ( v5 )
    v6 = *(_DWORD *)(v5 + 12);
  *(_DWORD *)(a1 + 20) = v6;
  return a1;
}
