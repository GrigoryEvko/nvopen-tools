// Function: sub_235B5E0
// Address: 0x235b5e0
//
_QWORD *__fastcall sub_235B5E0(unsigned __int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int16 v8; // r12
  _QWORD *result; // rax
  _QWORD *v10; // rdi
  char *v11; // rsi
  _QWORD *v12; // [rsp+8h] [rbp-28h] BYREF

  sub_2332320((__int64)a1, 1, a3, a4, a5, a6);
  v7 = *a2;
  v8 = *((_WORD *)a2 + 4);
  result = (_QWORD *)sub_22077B0(0x18u);
  v10 = result;
  if ( result )
  {
    result[1] = v7;
    *((_WORD *)result + 8) = v8;
    result = &unk_4A140B8;
    *v10 = &unk_4A140B8;
  }
  v12 = v10;
  v11 = (char *)a1[13];
  if ( v11 == (char *)a1[14] )
  {
    result = (_QWORD *)sub_235B010(a1 + 12, v11, &v12);
    v10 = v12;
  }
  else
  {
    if ( v11 )
    {
      *(_QWORD *)v11 = v10;
      a1[13] += 8LL;
      return result;
    }
    a1[13] = 8;
  }
  if ( v10 )
    return (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *))(*v10 + 8LL))(v10);
  return result;
}
