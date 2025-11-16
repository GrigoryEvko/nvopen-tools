// Function: sub_2252030
// Address: 0x2252030
//
unsigned __int64 __fastcall sub_2252030(__int64 *a1, const wchar_t *a2, __int64 a3)
{
  __int64 v4; // rdx
  size_t v5; // r12
  wchar_t *v6; // rdi
  unsigned __int64 result; // rax
  __int64 v8; // rax
  unsigned __int64 v9[4]; // [rsp+8h] [rbp-20h] BYREF

  if ( a3 && !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v4 = a3 - (_QWORD)a2;
  v5 = v4 >> 2;
  v9[0] = v4 >> 2;
  if ( (unsigned __int64)v4 > 0xC )
  {
    v8 = sub_22517A0((__int64)a1, v9, 0);
    *a1 = v8;
    v6 = (wchar_t *)v8;
    a1[2] = v9[0];
  }
  else
  {
    v6 = (wchar_t *)*a1;
  }
  if ( v5 == 1 )
  {
    *v6 = *a2;
  }
  else if ( v5 )
  {
    wmemcpy(v6, a2, v5);
    v6 = (wchar_t *)*a1;
  }
  result = v9[0];
  a1[1] = v9[0];
  v6[result] = 0;
  return result;
}
