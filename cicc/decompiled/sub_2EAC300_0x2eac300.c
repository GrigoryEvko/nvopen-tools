// Function: sub_2EAC300
// Address: 0x2eac300
//
__int64 __fastcall sub_2EAC300(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rax
  int v6; // edx

  v5 = sub_2F3F610(*(_QWORD *)(a2 + 352), a3);
  *(_QWORD *)(a1 + 8) = a4;
  *(_BYTE *)(a1 + 20) = 0;
  *(_QWORD *)a1 = v5 | 4;
  v6 = 0;
  if ( v5 )
    v6 = *(_DWORD *)(v5 + 12);
  *(_DWORD *)(a1 + 16) = v6;
  return a1;
}
