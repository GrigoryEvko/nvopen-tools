// Function: sub_890140
// Address: 0x890140
//
_QWORD *__fastcall sub_890140(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v8; // rdi
  _QWORD *v9; // r14
  _QWORD *result; // rax
  __int64 v11; // rax
  _QWORD v12[2]; // [rsp+0h] [rbp-40h] BYREF
  int v13; // [rsp+10h] [rbp-30h]

  v12[0] = a1;
  v8 = (unsigned __int8 *)a2[17];
  v12[1] = a4;
  v13 = 0;
  if ( !v8 )
  {
    v11 = sub_881A70(0, 0xBu, 12, 13, a5, a6);
    a2[17] = v11;
    v8 = (unsigned __int8 *)v11;
  }
  v9 = (_QWORD *)sub_881B20(v8, (__int64)v12, 1);
  result = (_QWORD *)*(unsigned __int8 *)(a1 + 80);
  if ( (_BYTE)result == 19 )
  {
    result = sub_878440();
    result[1] = a3;
    *result = a2[21];
    a2[21] = result;
    *v9 = a3;
  }
  else
  {
    if ( (_BYTE)result == 21 )
    {
      result = sub_878440();
      result[1] = a3;
      *result = a2[23];
      a2[23] = result;
    }
    *v9 = a3;
  }
  return result;
}
