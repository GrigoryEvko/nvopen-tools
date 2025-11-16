// Function: sub_DFCEF0
// Address: 0xdfcef0
//
__int64 __fastcall sub_DFCEF0(__int64 **a1, unsigned __int8 *a2, unsigned __int8 **a3, __int64 a4, int a5)
{
  __int64 *v5; // rdi
  __int64 (*v6)(void); // rax

  v5 = *a1;
  v6 = *(__int64 (**)(void))(*v5 + 120);
  if ( (char *)v6 == (char *)sub_DFCEE0 )
    return sub_DFBE30(v5 + 1, a2, a3, a4, a5);
  else
    return v6();
}
