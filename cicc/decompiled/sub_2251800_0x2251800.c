// Function: sub_2251800
// Address: 0x2251800
//
size_t __fastcall sub_2251800(__int64 *a1, unsigned __int64 a2, wchar_t a3)
{
  wchar_t *v5; // rdi
  size_t result; // rax
  __int64 v7; // rax
  size_t v8[4]; // [rsp+8h] [rbp-20h] BYREF

  v8[0] = a2;
  if ( a2 > 3 )
  {
    v7 = sub_22517A0((__int64)a1, v8, 0);
    *a1 = v7;
    v5 = (wchar_t *)v7;
    result = v8[0];
    a1[2] = v8[0];
  }
  else
  {
    v5 = (wchar_t *)*a1;
    result = a2;
  }
  if ( result )
  {
    if ( result == 1 )
    {
      *v5 = a3;
    }
    else
    {
      wmemset(v5, a3, result);
      result = v8[0];
      v5 = (wchar_t *)*a1;
    }
  }
  a1[1] = result;
  v5[result] = 0;
  return result;
}
