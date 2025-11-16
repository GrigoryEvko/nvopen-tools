// Function: sub_38DBF40
// Address: 0x38dbf40
//
void __noreturn sub_38DBF40()
{
  _QWORD *v0; // rax
  char *v1; // rdx

  v0 = sub_16E8CB0();
  v1 = (char *)v0[3];
  if ( v0[2] - (_QWORD)v1 > 0x61u )
  {
    qmemcpy(
      v1,
      "EmitRawText called on an MCStreamer that doesn't support it,  something must not be fully mc'ized\n",
      0x62u);
    v0[3] += 98LL;
  }
  else
  {
    sub_16E7EE0(
      (__int64)v0,
      "EmitRawText called on an MCStreamer that doesn't support it,  something must not be fully mc'ized\n",
      0x62u);
  }
  abort();
}
