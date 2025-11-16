// Function: sub_1E0CF70
// Address: 0x1e0cf70
//
char *__fastcall sub_1E0CF70(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, int a6)
{
  char *result; // rax
  _BYTE *v7; // rsi
  int v8; // [rsp+Ch] [rbp-4h] BYREF

  result = (char *)sub_1E0C9D0(a1, a2, a3, a4, a5, a6);
  v8 = 0;
  v7 = (_BYTE *)*((_QWORD *)result + 13);
  if ( v7 == *((_BYTE **)result + 14) )
    return sub_1E0CD40((__int64)(result + 96), v7, &v8);
  if ( v7 )
  {
    *(_DWORD *)v7 = 0;
    v7 = (_BYTE *)*((_QWORD *)result + 13);
  }
  *((_QWORD *)result + 13) = v7 + 4;
  return result;
}
