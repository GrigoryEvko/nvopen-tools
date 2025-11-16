// Function: sub_31F6FE0
// Address: 0x31f6fe0
//
unsigned __int64 *__fastcall sub_31F6FE0(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r13
  _QWORD *v5; // r15
  _QWORD v7[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(_QWORD **)(a2 + 16);
  v5 = *(_QWORD **)(a2 + 8);
  if ( v5 == v4 )
  {
LABEL_6:
    *a1 = 1;
  }
  else
  {
    while ( 1 )
    {
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*(_QWORD *)*v5 + 56LL))(v7, *v5, a3);
      if ( (v7[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      if ( v4 == ++v5 )
        goto LABEL_6;
    }
    *a1 = v7[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  return a1;
}
