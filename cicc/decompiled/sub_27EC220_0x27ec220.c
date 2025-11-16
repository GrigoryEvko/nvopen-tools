// Function: sub_27EC220
// Address: 0x27ec220
//
void __fastcall sub_27EC220(_QWORD *a1)
{
  __int64 v1; // rsi

  v1 = a1[9];
  *a1 = &off_4A21100;
  if ( v1 )
    sub_B91220((__int64)(a1 + 9), v1);
  j_j___libc_free_0((unsigned __int64)a1);
}
