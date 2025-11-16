// Function: sub_C92B90
// Address: 0xc92b90
//
__int64 __fastcall sub_C92B90(__int64 *a1, char *a2, __int64 (__fastcall *a3)(__int64), char *a4)
{
  char *v7; // rbx
  _BYTE *v8; // r15
  __int64 v9; // rdi
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a2;
  v12[0] = a4 - a2;
  if ( (unsigned __int64)(a4 - a2) > 0xF )
  {
    v11 = sub_22409D0(a1, v12, 0);
    *a1 = v11;
    v8 = (_BYTE *)v11;
    a1[2] = v12[0];
  }
  else
  {
    v8 = (_BYTE *)*a1;
  }
  if ( a4 != a2 )
  {
    do
    {
      v9 = (unsigned int)*v7++;
      *v8++ = a3(v9);
    }
    while ( a4 != v7 );
    v8 = (_BYTE *)*a1;
  }
  result = v12[0];
  a1[1] = v12[0];
  v8[result] = 0;
  return result;
}
