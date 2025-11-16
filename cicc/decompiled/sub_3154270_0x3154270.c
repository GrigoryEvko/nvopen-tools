// Function: sub_3154270
// Address: 0x3154270
//
__int64 *__fastcall sub_3154270(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 v5; // rax

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84053) )
  {
    v3 = *a2;
    *a2 = 0;
    *a1 = 1;
    if ( v3 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
    return a1;
  }
  else
  {
    v5 = *a2;
    *a2 = 0;
    *a1 = v5 | 1;
    return a1;
  }
}
