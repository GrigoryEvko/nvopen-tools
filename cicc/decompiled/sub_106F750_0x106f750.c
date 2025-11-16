// Function: sub_106F750
// Address: 0x106f750
//
__int64 __fastcall sub_106F750(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 (__fastcall *v3)(__int64); // rax

  v2 = a1[15];
  *a1 = &unk_49E6078;
  if ( v2 )
  {
    v3 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL);
    if ( v3 == sub_106F6B0 )
    {
      nullsub_397();
      a2 = 8;
      j_j___libc_free_0(v2, 8);
    }
    else
    {
      v3(v2);
    }
  }
  sub_E8EC10((__int64)a1, a2);
  return j_j___libc_free_0(a1, 128);
}
