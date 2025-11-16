// Function: sub_7FCC60
// Address: 0x7fcc60
//
__int64 *__fastcall sub_7FCC60(__int64 a1, __int64 a2, __int16 a3)
{
  __int64 *v4; // r12
  unsigned __int16 v5; // bx
  _QWORD *v7; // rax
  __int64 v8; // rsi

  if ( (_DWORD)a2 )
  {
    v4 = (__int64 *)sub_7E2510(a1, a2);
    v5 = a3 - 1;
    if ( !v5 )
      return v4;
  }
  else
  {
    v4 = sub_73E830(a1);
    v5 = a3 - 1;
    if ( !v5 )
      return v4;
  }
  v7 = sub_73A830(v5, byte_4F06A51[0]);
  v8 = *v4;
  v4[2] = (__int64)v7;
  return (__int64 *)sub_73DBF0(0x32u, v8, (__int64)v4);
}
