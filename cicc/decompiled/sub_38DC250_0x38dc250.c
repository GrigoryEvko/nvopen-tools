// Function: sub_38DC250
// Address: 0x38dc250
//
void (*__fastcall sub_38DC250(_QWORD *a1, unsigned __int64 a2, unsigned int a3))()
{
  char v3; // r9
  __int64 v4; // rax
  char v5; // cl
  void (*result)(); // rax
  _BYTE v7[8]; // [rsp+8h] [rbp-8h] BYREF

  v3 = *(_BYTE *)(*(_QWORD *)(a1[1] + 16LL) + 16LL);
  if ( a3 )
  {
    v4 = 0;
    do
    {
      v5 = a3 - 1 - v4;
      if ( v3 )
        v5 = v4;
      v7[v4++] = a2 >> (8 * v5);
    }
    while ( a3 != (_DWORD)v4 );
  }
  result = *(void (**)())(*a1 + 400LL);
  if ( result != nullsub_1953 )
    return (void (*)())((__int64 (__fastcall *)(_QWORD *, _BYTE *, _QWORD))result)(a1, v7, a3);
  return result;
}
