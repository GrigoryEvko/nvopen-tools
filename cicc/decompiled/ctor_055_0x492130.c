// Function: ctor_055
// Address: 0x492130
//
int ctor_055()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  int *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v1 = 100;
  v2 = &v1;
  v3[0] = "Maximum number of instructions for ObjectSizeOffsetVisitor to look at";
  v3[1] = 69;
  sub_D5F340(&unk_4F87740, "object-size-offset-visitor-max-visit-instructions", v3, &v2);
  return __cxa_atexit(sub_984970, &unk_4F87740, &qword_4A427C0);
}
