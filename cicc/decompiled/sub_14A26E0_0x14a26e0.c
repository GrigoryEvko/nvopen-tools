// Function: sub_14A26E0
// Address: 0x14a26e0
//
__int64 __fastcall sub_14A26E0(__int64 **a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rdi
  __int64 (*v6)(void); // rax

  v5 = *a1;
  v6 = *(__int64 (**)(void))(*v5 + 32);
  if ( (char *)v6 == (char *)sub_14A1CD0 )
    return sub_14A1310(v5 + 1, a2, a3, a4, a5);
  else
    return v6();
}
