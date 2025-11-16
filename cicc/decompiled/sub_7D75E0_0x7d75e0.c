// Function: sub_7D75E0
// Address: 0x7d75e0
//
__int64 __fastcall sub_7D75E0(unsigned __int8 a1, const char *a2)
{
  __int64 result; // rax
  _QWORD *v3; // r14
  size_t v4; // rax
  char *v5; // rax

  result = sub_72C7B0(a1);
  if ( (_DWORD)result )
  {
    v3 = sub_72C7D0(a1);
    sub_725570((__int64)v3, 12);
    v3[20] = sub_72C610(a1);
    v4 = strlen(a2);
    v5 = (char *)sub_7247C0(v4 + 1);
    v3[1] = v5;
    strcpy(v5, a2);
    return sub_7E1CA0(v3);
  }
  return result;
}
