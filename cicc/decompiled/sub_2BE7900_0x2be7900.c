// Function: sub_2BE7900
// Address: 0x2be7900
//
_BYTE *__fastcall sub_2BE7900(_QWORD *a1, __int64 a2)
{
  char v2; // r12
  _BYTE *v3; // rax
  _BYTE *result; // rax
  __int64 v5; // r13
  unsigned int v6; // r14d
  __int64 v7; // rax
  char v8; // al
  char *v9; // rsi
  char v10[33]; // [rsp+Fh] [rbp-21h] BYREF

  v2 = a2;
  v3 = (_BYTE *)*a1;
  if ( *(_BYTE *)*a1 )
  {
    v5 = a1[1];
    v6 = (char)v3[1];
    v7 = sub_222F790(*(_QWORD **)(v5 + 104), a2);
    v8 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v7 + 32LL))(v7, v6);
    v9 = *(char **)(v5 + 8);
    v10[0] = v8;
    if ( v9 == *(char **)(v5 + 16) )
    {
      sub_17EB120(v5, v9, v10);
    }
    else
    {
      if ( v9 )
      {
        *v9 = v8;
        v9 = *(char **)(v5 + 8);
      }
      *(_QWORD *)(v5 + 8) = v9 + 1;
    }
  }
  else
  {
    *v3 = 1;
  }
  result = (_BYTE *)*a1;
  *(_BYTE *)(*a1 + 1LL) = v2;
  return result;
}
