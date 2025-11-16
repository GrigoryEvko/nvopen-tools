// Function: sub_1F4E280
// Address: 0x1f4e280
//
char *__fastcall sub_1F4E280(__int64 a1, int a2, __int64 a3, unsigned __int8 a4, __int64 a5, unsigned __int64 a6)
{
  char *result; // rax
  _BYTE *v8; // rsi
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  result = (char *)sub_1E1AFE0(a3, a2, *(_QWORD **)(a1 + 360), a4, a5, a6);
  if ( (_BYTE)result )
  {
    result = sub_1DCC790((char *)a1, a2);
    v9[0] = a3;
    v8 = (_BYTE *)*((_QWORD *)result + 5);
    if ( v8 == *((_BYTE **)result + 6) )
    {
      return sub_1DCC370((__int64)(result + 32), v8, v9);
    }
    else
    {
      if ( v8 )
      {
        *(_QWORD *)v8 = a3;
        v8 = (_BYTE *)*((_QWORD *)result + 5);
      }
      *((_QWORD *)result + 5) = v8 + 8;
    }
  }
  return result;
}
