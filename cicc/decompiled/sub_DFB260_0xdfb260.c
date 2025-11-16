// Function: sub_DFB260
// Address: 0xdfb260
//
__int64 __fastcall sub_DFB260(__int64 a1)
{
  __int64 (*v1)(void); // rax
  __int64 v3; // [rsp-8h] [rbp-8h]

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1032LL);
  if ( v1 != sub_DF6040 )
    return v1();
  *((_BYTE *)&v3 - 4) = 0;
  return *(&v3 - 1);
}
