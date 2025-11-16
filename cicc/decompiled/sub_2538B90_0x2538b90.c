// Function: sub_2538B90
// Address: 0x2538b90
//
__int64 __fastcall sub_2538B90(__int64 a1)
{
  __int64 (__fastcall *v1)(__int64); // rax

  v1 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(a1 - 104) + 112LL);
  if ( v1 == sub_2534E30 )
    return *(_QWORD *)(a1 + 48);
  else
    return *(_QWORD *)(v1(a1 - 104) + 32);
}
