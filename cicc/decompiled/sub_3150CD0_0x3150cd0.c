// Function: sub_3150CD0
// Address: 0x3150cd0
//
_BYTE *__fastcall sub_3150CD0(unsigned __int64 *a1, char *a2)
{
  char v3; // bl
  _BYTE *result; // rax
  _BYTE *v5; // rdi
  char *v6; // rsi
  _BYTE *v7; // [rsp+8h] [rbp-18h] BYREF

  v3 = *a2;
  result = (_BYTE *)sub_22077B0(0x10u);
  v5 = result;
  if ( result )
  {
    result[8] = v3;
    result = &unk_4A0EC78;
    *(_QWORD *)v5 = &unk_4A0EC78;
  }
  v7 = v5;
  v6 = (char *)a1[1];
  if ( v6 == (char *)a1[2] )
  {
    result = (_BYTE *)sub_235A6C0(a1, v6, &v7);
    v5 = v7;
  }
  else
  {
    if ( v6 )
    {
      *(_QWORD *)v6 = v5;
      a1[1] += 8LL;
      return result;
    }
    a1[1] = 8;
  }
  if ( v5 )
    return (_BYTE *)(*(__int64 (__fastcall **)(_BYTE *))(*(_QWORD *)v5 + 8LL))(v5);
  return result;
}
