// Function: sub_334D620
// Address: 0x334d620
//
_QWORD *__fastcall sub_334D620(_QWORD *a1, const char *a2, const char *a3, __int64 a4)
{
  size_t v5; // r14
  size_t v8; // r9
  _QWORD *result; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi

  v5 = 0;
  *a1 = 0;
  a1[1] = a2;
  if ( a2 )
    v5 = strlen(a2);
  a1[2] = v5;
  v8 = 0;
  a1[3] = a3;
  if ( a3 )
    v8 = strlen(a3);
  result = qword_5039AB0;
  a1[4] = v8;
  a1[5] = a4;
  v10 = qword_5039AB0[0];
  v11 = unk_5039AC0;
  qword_5039AB0[0] = a1;
  *a1 = v10;
  if ( v11 )
    return (_QWORD *)(*(__int64 (__fastcall **)(__int64, const char *, size_t, __int64, const char *))(*(_QWORD *)v11 + 24LL))(
                       v11,
                       a2,
                       v5,
                       a4,
                       a3);
  return result;
}
