// Function: sub_235B530
// Address: 0x235b530
//
_DWORD *__fastcall sub_235B530(unsigned __int64 *a1, int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // r12d
  _DWORD *result; // rax
  _DWORD *v9; // rdi
  char *v10; // rsi
  _DWORD *v11; // [rsp+8h] [rbp-18h] BYREF

  sub_2332320((__int64)a1, 1, a3, a4, a5, a6);
  v7 = *a2;
  result = (_DWORD *)sub_22077B0(0x10u);
  v9 = result;
  if ( result )
  {
    result[2] = v7;
    result = &unk_4A13DB8;
    *(_QWORD *)v9 = &unk_4A13DB8;
  }
  v11 = v9;
  v10 = (char *)a1[13];
  if ( v10 == (char *)a1[14] )
  {
    result = (_DWORD *)sub_235B010(a1 + 12, v10, &v11);
    v9 = v11;
  }
  else
  {
    if ( v10 )
    {
      *(_QWORD *)v10 = v9;
      a1[13] += 8LL;
      return result;
    }
    a1[13] = 8;
  }
  if ( v9 )
    return (_DWORD *)(*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v9 + 8LL))(v9);
  return result;
}
