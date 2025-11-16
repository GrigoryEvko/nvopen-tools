// Function: ctor_542
// Address: 0x56bda0
//
int ctor_542()
{
  int v1; // [rsp+0h] [rbp-20h] BYREF
  int v2; // [rsp+4h] [rbp-1Ch] BYREF
  void *v3; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v4[2]; // [rsp+10h] [rbp-10h] BYREF

  v4[1] = 39;
  v3 = &unk_50165C8;
  v4[0] = "Aggresive floating point simplification";
  v1 = 1;
  v2 = 1;
  ((void (__fastcall *)(void *, const char *, int *, int *, _QWORD *, void **))sub_2D22540)(
    &unk_5016500,
    "opt-unsafe-algebra",
    &v2,
    &v1,
    v4,
    &v3);
  return __cxa_atexit(sub_AA4490, &unk_5016500, &qword_4A427C0);
}
