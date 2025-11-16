// Function: sub_E46790
// Address: 0xe46790
//
__int64 *__fastcall sub_E46790(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // r14
  __int64 v6; // rax

  if ( (*(unsigned __int8 (__fastcall **)(__int64, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84053) )
  {
    v4 = *a2;
    *a2 = 0;
    sub_E46480(a3, v4);
    *a1 = 1;
    if ( v4 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
    return a1;
  }
  else
  {
    v6 = *a2;
    *a2 = 0;
    *a1 = v6 | 1;
    return a1;
  }
}
