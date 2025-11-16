// Function: sub_DF9D20
// Address: 0xdf9d20
//
__int64 __fastcall sub_DF9D20(__int64 a1)
{
  __int64 (*v1)(void); // rax
  __int64 v3; // [rsp-8h] [rbp-8h]

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 408LL);
  if ( v1 != sub_DF5DB0 )
    return v1();
  *((_BYTE *)&v3 - 8) = 0;
  return *(&v3 - 2);
}
