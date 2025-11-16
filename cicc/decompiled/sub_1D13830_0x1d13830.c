// Function: sub_1D13830
// Address: 0x1d13830
//
void __fastcall sub_1D13830(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 (*v3)(); // rax
  _QWORD v4[2]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v5; // [rsp-38h] [rbp-38h]
  __int64 v6; // [rsp-28h] [rbp-28h]
  __int64 v7; // [rsp-8h] [rbp-8h]

  if ( (_DWORD)a2 )
  {
    v7 = v2;
    v3 = *(__int64 (**)())(*(_QWORD *)a1 + 576LL);
    if ( v3 == sub_1D12D90 || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v3)(a1, a2, 0) )
    {
      LODWORD(v6) = a2;
      v4[0] = "cannot lower memory intrinsic in address space ";
      v4[1] = v6;
      v5 = 2307;
      sub_16BCFB0((__int64)v4, 1u);
    }
  }
}
