// Function: sub_3704000
// Address: 0x3704000
//
__int64 *__fastcall sub_3704000(__int64 *a1, __int64 *a2, unsigned __int64 a3, const void *a4, size_t a5)
{
  __int64 (__fastcall *v9)(__int64); // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  int v14[13]; // [rsp+Ch] [rbp-34h] BYREF

  if ( a5 )
  {
    v9 = *(__int64 (__fastcall **)(__int64))(*a2 + 40);
    if ( v9 == sub_3702AB0 )
      v10 = a2[2] - a2[1];
    else
      v10 = v9((__int64)a2);
    if ( v10 < a3 )
    {
      v14[0] = 3;
      sub_3703E00(a1, v14);
      return a1;
    }
    v11 = a2[1];
    v12 = a2[2] - v11;
    if ( a5 + a3 > v12 )
    {
      sub_CD93F0(a2 + 1, a5 + a3 - v12);
      v11 = a2[1];
    }
    memcpy((void *)(a3 + v11), a4, a5);
  }
  *a1 = 1;
  return a1;
}
