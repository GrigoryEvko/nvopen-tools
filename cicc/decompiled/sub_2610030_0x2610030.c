// Function: sub_2610030
// Address: 0x2610030
//
void __fastcall sub_2610030(_QWORD *a1)
{
  __int64 v1; // rsi

  v1 = a1[4];
  *a1 = &unk_4A1F3E0;
  if ( v1 )
    sub_B91220((__int64)(a1 + 4), v1);
  j_j___libc_free_0((unsigned __int64)a1);
}
