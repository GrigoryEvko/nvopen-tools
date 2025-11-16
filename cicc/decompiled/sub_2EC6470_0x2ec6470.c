// Function: sub_2EC6470
// Address: 0x2ec6470
//
__int64 __fastcall sub_2EC6470(__int64 a1)
{
  __int64 *v1; // r13
  __int64 *i; // rbx
  __int64 v4; // rdi
  __int64 result; // rax

  v1 = *(__int64 **)(a1 + 3488);
  for ( i = *(__int64 **)(a1 + 3480);
        v1 != i;
        result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 24LL))(v4, a1) )
  {
    v4 = *i++;
  }
  return result;
}
