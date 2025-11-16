// Function: sub_C3C460
// Address: 0xc3c460
//
_QWORD *__fastcall sub_C3C460(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // r15
  _QWORD *v4; // r12
  _DWORD *v5; // rax
  _DWORD *v6; // r14

  *a1 = a2;
  result = (_QWORD *)sub_2207820(56);
  if ( result )
  {
    *result = 2;
    v3 = result + 1;
    v4 = result + 4;
    v5 = sub_C33340();
    v6 = v5;
    if ( v5 == dword_3F657A0 )
    {
      sub_C3C460(v3, v5);
      result = (_QWORD *)sub_C3C460(v4, v6);
    }
    else
    {
      sub_C37380(v3, (__int64)dword_3F657A0);
      result = (_QWORD *)sub_C37380(v4, (__int64)dword_3F657A0);
    }
  }
  else
  {
    v3 = 0;
  }
  a1[1] = v3;
  return result;
}
