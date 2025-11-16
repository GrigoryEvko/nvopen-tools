// Function: sub_31F5EB0
// Address: 0x31f5eb0
//
unsigned __int64 *__fastcall sub_31F5EB0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // r14
  _QWORD *v7; // r13
  _QWORD v9[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_QWORD **)(a2 + 16);
  v7 = *(_QWORD **)(a2 + 8);
  if ( v7 == v6 )
  {
LABEL_6:
    *a1 = 1;
  }
  else
  {
    while ( 1 )
    {
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64, __int64))(*(_QWORD *)*v7 + 256LL))(v9, *v7, a3, a4);
      if ( (v9[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      if ( v6 == ++v7 )
        goto LABEL_6;
    }
    *a1 = v9[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  return a1;
}
