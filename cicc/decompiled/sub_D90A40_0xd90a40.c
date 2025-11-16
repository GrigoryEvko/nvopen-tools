// Function: sub_D90A40
// Address: 0xd90a40
//
__int64 __fastcall sub_D90A40(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 *v3; // rbx
  __int64 result; // rax
  __int64 *i; // r14
  __int64 v7; // rdi

  v3 = *(__int64 **)(a1 + 40);
  result = *(unsigned int *)(a1 + 48);
  for ( i = &v3[result];
        i != v3;
        result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v7 + 24LL))(v7, a2, a3) )
  {
    v7 = *v3++;
  }
  return result;
}
