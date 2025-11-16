// Function: sub_F03820
// Address: 0xf03820
//
__int64 __fastcall sub_F03820(__int64 a1, int a2)
{
  char *v3; // r13
  size_t v4; // rax
  char buf[2016]; // [rsp+0h] [rbp-7E0h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  if ( a2 )
  {
    buf[0] = 0;
    v3 = strerror_r(a2, buf, 0x7CFu);
    v4 = strlen(v3);
    sub_2241130(a1, 0, *(_QWORD *)(a1 + 8), v3, v4);
  }
  return a1;
}
