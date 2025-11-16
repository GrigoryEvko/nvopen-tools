// Function: sub_2538C30
// Address: 0x2538c30
//
__int64 __fastcall sub_2538C30(__int64 a1)
{
  __int64 (__fastcall *v1)(__int64); // rax
  __int64 v2; // rax

  v1 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(a1 - 104) + 112LL);
  if ( v1 == sub_2534E30 )
    v2 = a1 + 16;
  else
    v2 = v1(a1 - 104);
  return *(_QWORD *)(v2 + 32) + 8LL * *(unsigned int *)(v2 + 40);
}
