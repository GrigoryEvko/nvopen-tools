// Function: sub_DF9500
// Address: 0xdf9500
//
__int64 __fastcall sub_DF9500(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rdi
  __int64 (*v6)(void); // rax

  v5 = *a1;
  v6 = *(__int64 (**)(void))(*v5 + 24);
  if ( (char *)v6 == (char *)sub_DF79A0 )
    return sub_DF7390(v5 + 1, a2, a3, a4, a5);
  else
    return v6();
}
