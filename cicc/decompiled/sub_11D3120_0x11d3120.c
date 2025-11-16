// Function: sub_11D3120
// Address: 0x11d3120
//
void *__fastcall sub_11D3120(_QWORD *a1, _BYTE **a2, __int64 a3, __int64 *a4, unsigned __int8 *a5, size_t a6)
{
  void *result; // rax
  _BYTE *v8; // r12
  __int64 *v9; // rdi
  const char *v10; // rax
  size_t v11; // rdx

  result = &unk_49E63B0;
  *a1 = &unk_49E63B0;
  a1[1] = a4;
  if ( a3 )
  {
    v8 = *a2;
    v9 = a4;
    if ( **a2 == 61 )
    {
      if ( a6 )
        return (void *)sub_11D2C80(v9, *((_QWORD *)v8 + 1), a5, a6);
    }
    else
    {
      v8 = (_BYTE *)*((_QWORD *)v8 - 8);
      if ( a6 )
        return (void *)sub_11D2C80(v9, *((_QWORD *)v8 + 1), a5, a6);
    }
    v10 = sub_BD5D20((__int64)v8);
    v9 = (__int64 *)a1[1];
    a5 = (unsigned __int8 *)v10;
    a6 = v11;
    return (void *)sub_11D2C80(v9, *((_QWORD *)v8 + 1), a5, a6);
  }
  return result;
}
