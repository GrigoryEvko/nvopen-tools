// Function: ctor_631
// Address: 0x591c80
//
int ctor_631()
{
  _BYTE v1[17]; // [rsp+Fh] [rbp-11h] BYREF

  sub_A758C0(&unk_5031DE0, "INVALID", v1);
  sub_A758C0(&unk_5031E00, "float", v1);
  sub_A758C0(&unk_5031E20, "double", v1);
  sub_A758C0(&unk_5031E40, "int8_t", v1);
  sub_A758C0(&unk_5031E60, "uint8_t", v1);
  sub_A758C0(&unk_5031E80, "int16_t", v1);
  sub_A758C0(&unk_5031EA0, "uint16_t", v1);
  sub_A758C0(&unk_5031EC0, "int32_t", v1);
  sub_A758C0(&unk_5031EE0, "uint32_t", v1);
  sub_A758C0(&unk_5031F00, "int64_t", v1);
  sub_A758C0(&unk_5031F20, "uint64_t", v1);
  return __cxa_atexit(sub_310CFB0, &unk_5031DE0, &qword_4A427C0);
}
