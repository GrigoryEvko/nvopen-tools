// Function: sub_C63DA0
// Address: 0xc63da0
//
__int64 *__fastcall sub_C63DA0(__int64 *a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v5; // r14
  __int64 v6; // rdi
  _BYTE *v7; // rax
  __int64 v9; // rax

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84053) )
  {
    v5 = *a2;
    *a2 = 0;
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v5 + 16LL))(v5, *a3);
    v6 = *a3;
    v7 = *(_BYTE **)(*a3 + 32LL);
    if ( *(_BYTE **)(*a3 + 24LL) == v7 )
    {
      sub_CB6200(v6, "\n", 1);
    }
    else
    {
      *v7 = 10;
      ++*(_QWORD *)(v6 + 32);
    }
    *a1 = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
    return a1;
  }
  else
  {
    v9 = *a2;
    *a2 = 0;
    *a1 = v9 | 1;
    return a1;
  }
}
