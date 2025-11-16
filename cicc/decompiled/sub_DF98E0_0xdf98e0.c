// Function: sub_DF98E0
// Address: 0xdf98e0
//
__int64 __fastcall sub_DF98E0(__int64 a1)
{
  __int64 (*v1)(void); // rax
  __int64 v3; // [rsp-8h] [rbp-8h]

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 208LL);
  if ( v1 != sub_DF5C90 )
    return v1();
  *((_BYTE *)&v3 - 4) = 0;
  return *(__int64 *)((char *)&v3 - 12);
}
