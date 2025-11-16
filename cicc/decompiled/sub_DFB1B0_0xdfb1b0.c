// Function: sub_DFB1B0
// Address: 0xdfb1b0
//
__int64 __fastcall sub_DFB1B0(__int64 a1)
{
  __int64 (*v1)(void); // rax
  __int64 v3; // [rsp-8h] [rbp-8h]

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1008LL);
  if ( v1 != sub_DF6010 )
    return v1();
  *(&v3 - 2) = 32;
  *((_BYTE *)&v3 - 8) = 0;
  return *(&v3 - 2);
}
