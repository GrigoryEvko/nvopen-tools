// Function: sub_C63D10
// Address: 0xc63d10
//
__int64 *__fastcall sub_C63D10(__int64 *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v8; // rax

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84053) )
  {
    v4 = *a2;
    *a2 = 0;
    v5 = *a3;
    *(_DWORD *)v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 32LL))(v4);
    *(_QWORD *)(v5 + 8) = v6;
    *a1 = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  }
  else
  {
    v8 = *a2;
    *a2 = 0;
    *a1 = v8 | 1;
  }
  return a1;
}
