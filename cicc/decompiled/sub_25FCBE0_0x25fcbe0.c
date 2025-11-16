// Function: sub_25FCBE0
// Address: 0x25fcbe0
//
unsigned __int64 __fastcall sub_25FCBE0(__int64 a1, __int64 **a2)
{
  __int64 *v2; // r15
  unsigned __int64 v3; // r12
  __int64 v4; // rbx
  __int64 **v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  signed __int64 v10; // rax
  bool v11; // of
  __int64 *v13; // [rsp+8h] [rbp-38h]

  v13 = a2[1];
  if ( *a2 == v13 )
    return 0;
  v2 = *a2;
  v3 = 0;
  do
  {
    v4 = *v2;
    v5 = (__int64 **)(*(__int64 (__fastcall **)(_QWORD, _QWORD))(a1 + 40))(
                       *(_QWORD *)(a1 + 48),
                       *(_QWORD *)(*(_QWORD *)(*v2 + 272) + 72LL));
    v10 = sub_25FBC60(v4, v5, v6, v7, v8, v9);
    v11 = __OFADD__(v10, v3);
    v3 += v10;
    if ( v11 )
    {
      v3 = 0x8000000000000000LL;
      if ( v10 > 0 )
        v3 = 0x7FFFFFFFFFFFFFFFLL;
    }
    ++v2;
  }
  while ( v13 != v2 );
  return v3;
}
