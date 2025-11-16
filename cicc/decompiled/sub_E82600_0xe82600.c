// Function: sub_E82600
// Address: 0xe82600
//
__int64 __fastcall sub_E82600(unsigned int *a1, __int64 a2, __int64 a3, void *a4, size_t a5, __int64 a6)
{
  char *v6; // r10
  size_t v9; // rcx
  size_t v11; // rdx
  __int64 v13; // [rsp-10h] [rbp-40h]

  v6 = (char *)byte_3F871B3;
  v9 = 0;
  if ( a3 )
  {
    v6 = (char *)sub_E826A0(a3, *a1);
    v9 = v11;
  }
  sub_E82420(a1, a2, v6, v9, a4, a5, a6);
  return v13;
}
