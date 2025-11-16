// Function: sub_7D7C70
// Address: 0x7d7c70
//
__int64 __fastcall sub_7D7C70(char a1, const char *a2)
{
  __int64 result; // rax
  _QWORD *v3; // r14
  __int64 v4; // r12
  size_t v5; // rax
  char *v6; // rax

  result = sub_72C6D0(a1);
  if ( (_DWORD)result )
  {
    v3 = sub_72C6F0(a1);
    v4 = sub_7D7990(a1);
    sub_725570((__int64)v3, 12);
    v5 = strlen(a2);
    v6 = (char *)sub_7247C0(v5 + 1);
    v3[1] = v6;
    strcpy(v6, a2);
    *((_BYTE *)v3 + 186) |= 0x80u;
    v3[20] = v4;
    if ( (v3[11] & 8) != 0 )
    {
      sub_7604D0(v4, 6u);
      sub_75C030(v4);
      sub_75BF90(v4);
    }
    result = sub_7E1CA0(v3);
    if ( !*(_QWORD *)(v4 + 112) )
      return sub_7E1CA0(v4);
  }
  return result;
}
