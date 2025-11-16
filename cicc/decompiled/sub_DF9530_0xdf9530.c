// Function: sub_DF9530
// Address: 0xdf9530
//
unsigned __int64 __fastcall sub_DF9530(__int64 **a1, __int64 *a2, __int64 a3, _BYTE *a4, _BYTE *a5, __int64 a6, int a7)
{
  __int64 *v7; // rdi
  __int64 (*v8)(void); // rax

  v7 = *a1;
  v8 = *(__int64 (**)(void))(*v7 + 32);
  if ( (char *)v8 == (char *)sub_DF7C20 )
    return sub_DF79B0(v7 + 1, a2, a3, a4, a5, a6, a7);
  else
    return v8();
}
