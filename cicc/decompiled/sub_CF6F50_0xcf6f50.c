// Function: sub_CF6F50
// Address: 0xcf6f50
//
_QWORD *__fastcall sub_CF6F50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  void (__fastcall **v7)(__int64, __int64, _QWORD *); // rbx
  __int64 v8; // r14

  v6 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  sub_CF4B40(a1, v6 + 8);
  v7 = *(void (__fastcall ***)(__int64, __int64, _QWORD *))a2;
  v8 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( v8 != *(_QWORD *)a2 )
  {
    do
      (*v7++)(a3, a4, a1);
    while ( (void (__fastcall **)(__int64, __int64, _QWORD *))v8 != v7 );
  }
  return a1;
}
