// Function: sub_C7EC60
// Address: 0xc7ec60
//
char *__fastcall sub_C7EC60(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  char *(*v4)(); // rax
  char *result; // rax
  __int64 v6; // rdx

  v2 = a2[1];
  v3 = a2[2] - v2;
  *a1 = v2;
  a1[1] = v3;
  v4 = *(char *(**)())(*a2 + 16LL);
  if ( v4 == sub_C1E8B0 )
  {
    a1[3] = 14;
    a1[2] = "Unknown buffer";
    return "Unknown buffer";
  }
  else
  {
    result = (char *)((__int64 (__fastcall *)(_QWORD *))v4)(a2);
    a1[2] = result;
    a1[3] = v6;
  }
  return result;
}
