// Function: sub_223ED70
// Address: 0x223ed70
//
_QWORD *__fastcall sub_223ED70(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = *(_QWORD *)((char *)a1 + *(_QWORD *)(*a1 - 24LL) + 232);
  if ( !v2 || (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2) != -1 )
    return a1;
  sub_222DDD0((__int64)a1 + *(_QWORD *)(*a1 - 24LL), *(_DWORD *)((char *)a1 + *(_QWORD *)(*a1 - 24LL) + 32) | 1);
  return a1;
}
