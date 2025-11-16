// Function: sub_2F3A4B0
// Address: 0x2f3a4b0
//
void __fastcall sub_2F3A4B0(_QWORD *a1)
{
  __int64 v2; // rdi
  void (*v3)(void); // rdx
  void (*v4)(void); // rax
  _BYTE *v5; // rsi
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  v2 = a1[445];
  v3 = *(void (**)(void))(*(_QWORD *)v2 + 96LL);
  if ( (char *)v3 == (char *)sub_2F39570 )
  {
    v4 = *(void (**)(void))(*(_QWORD *)v2 + 80LL);
    if ( v4 != nullsub_1620 )
      v4();
  }
  else
  {
    v3();
  }
  v6 = 0;
  v5 = (_BYTE *)a1[449];
  if ( v5 == (_BYTE *)a1[450] )
  {
    sub_2F3A320((__int64)(a1 + 448), v5, &v6);
  }
  else
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = 0;
      v5 = (_BYTE *)a1[449];
    }
    a1[449] = v5 + 8;
  }
}
