// Function: sub_DFB220
// Address: 0xdfb220
//
__int64 __fastcall sub_DFB220(__int64 a1)
{
  __int64 (*v1)(void); // rax
  __int64 v3; // [rsp-8h] [rbp-8h]

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1024LL);
  if ( v1 != sub_DF6040 )
    return v1();
  *((_BYTE *)&v3 - 4) = 0;
  return *(&v3 - 1);
}
