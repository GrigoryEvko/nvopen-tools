// Function: sub_1F46420
// Address: 0x1f46420
//
void __fastcall sub_1F46420(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  void (__fastcall *v3)(__int64, __int64, _QWORD); // rbx
  __int64 v4; // rax

  if ( byte_4FCC9A0 )
  {
    v2 = *(_QWORD *)(a1 + 160);
    v3 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v2 + 16LL);
    v4 = sub_1E86650(a2);
    v3(v2, v4, 0);
  }
}
