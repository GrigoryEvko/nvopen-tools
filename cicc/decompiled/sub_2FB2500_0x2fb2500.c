// Function: sub_2FB2500
// Address: 0x2fb2500
//
__int64 __fastcall sub_2FB2500(__int64 a1)
{
  __int64 v2; // rdi
  int v3; // edx
  int v4; // eax

  v2 = *(_QWORD *)(a1 + 72);
  v3 = *(_DWORD *)(v2 + 64);
  v4 = *(_DWORD *)(*(_QWORD *)(v2 + 16) + 8LL);
  if ( v4 == v3 )
  {
    sub_350B2D0(v2, *(unsigned int *)(*(_QWORD *)(v2 + 8) + 112LL), 1);
    v2 = *(_QWORD *)(a1 + 72);
    v3 = *(_DWORD *)(v2 + 64);
    v4 = *(_DWORD *)(*(_QWORD *)(v2 + 16) + 8LL);
  }
  *(_DWORD *)(a1 + 80) = v4 - v3;
  sub_350B2D0(v2, *(unsigned int *)(*(_QWORD *)(v2 + 8) + 112LL), 1);
  return *(unsigned int *)(a1 + 80);
}
