// Function: sub_23A47E0
// Address: 0x23a47e0
//
_QWORD *__fastcall sub_23A47E0(unsigned __int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int16 v8; // r12
  _QWORD *result; // rax
  _QWORD *v10; // rdi
  char *v11; // rsi
  _QWORD *v12; // [rsp+8h] [rbp-28h] BYREF

  sub_2332320((__int64)a1, 0, a3, a4, a5, a6);
  v7 = *a2;
  v8 = *((_WORD *)a2 + 4);
  result = (_QWORD *)sub_22077B0(0x18u);
  v10 = result;
  if ( result )
  {
    result[1] = v7;
    *((_WORD *)result + 8) = v8;
    result = &unk_4A12478;
    *v10 = &unk_4A12478;
  }
  v12 = v10;
  v11 = (char *)a1[10];
  if ( v11 == (char *)a1[11] )
  {
    result = (_QWORD *)sub_235AB20(a1 + 9, v11, &v12);
    v10 = v12;
  }
  else
  {
    if ( v11 )
    {
      *(_QWORD *)v11 = v10;
      a1[10] += 8LL;
      return result;
    }
    a1[10] = 8;
  }
  if ( v10 )
    return (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *))(*v10 + 8LL))(v10);
  return result;
}
