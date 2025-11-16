// Function: sub_2E31540
// Address: 0x2e31540
//
__int64 __fastcall sub_2E31540(__int64 a1)
{
  unsigned __int64 v1; // r12
  __int64 (*v2)(); // rax
  __int64 v3; // rdi
  __int64 (*v4)(); // rax

  v1 = sub_2E313E0(a1);
  if ( v1 == a1 + 48 )
    return 0xFFFFFFFFLL;
  v2 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 32) + 16LL) + 128LL);
  if ( v2 == sub_2DAC790 )
    BUG();
  v3 = v2();
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 512LL);
  if ( v4 == sub_2E2F9A0 )
    return 0xFFFFFFFFLL;
  else
    return ((__int64 (__fastcall *)(__int64, unsigned __int64))v4)(v3, v1);
}
