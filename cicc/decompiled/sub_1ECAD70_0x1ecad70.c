// Function: sub_1ECAD70
// Address: 0x1ecad70
//
__int64 __fastcall sub_1ECAD70(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *i; // r13
  __int64 v4; // rdi
  __int64 result; // rax

  v2 = *(__int64 **)(a1 + 8);
  for ( i = *(__int64 **)(a1 + 16);
        i != v2;
        result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 16LL))(v4, a2) )
  {
    v4 = *v2++;
  }
  return result;
}
