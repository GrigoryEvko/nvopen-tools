// Function: sub_1E78570
// Address: 0x1e78570
//
__int64 __fastcall sub_1E78570(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi

  v2 = *(_QWORD *)(a1 + 464);
  *(_DWORD *)(a1 + 312) = 0;
  while ( v2 )
  {
    sub_1E783A0(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3, 48);
  }
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = a1 + 456;
  *(_QWORD *)(a1 + 480) = a1 + 456;
  *(_QWORD *)(a1 + 488) = 0;
  return a1 + 456;
}
