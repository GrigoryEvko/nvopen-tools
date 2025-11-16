// Function: sub_23A4730
// Address: 0x23a4730
//
_WORD *__fastcall sub_23A4730(unsigned __int64 *a1, __int16 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int16 v7; // r12
  _WORD *result; // rax
  _WORD *v9; // rdi
  char *v10; // rsi
  _WORD *v11; // [rsp+8h] [rbp-18h] BYREF

  sub_2332320((__int64)a1, 0, a3, a4, a5, a6);
  v7 = *a2;
  result = (_WORD *)sub_22077B0(0x10u);
  v9 = result;
  if ( result )
  {
    result[4] = v7;
    result = &unk_4A124F8;
    *(_QWORD *)v9 = &unk_4A124F8;
  }
  v11 = v9;
  v10 = (char *)a1[10];
  if ( v10 == (char *)a1[11] )
  {
    result = (_WORD *)sub_235AB20(a1 + 9, v10, &v11);
    v9 = v11;
  }
  else
  {
    if ( v10 )
    {
      *(_QWORD *)v10 = v9;
      a1[10] += 8LL;
      return result;
    }
    a1[10] = 8;
  }
  if ( v9 )
    return (_WORD *)(*(__int64 (__fastcall **)(_WORD *))(*(_QWORD *)v9 + 8LL))(v9);
  return result;
}
