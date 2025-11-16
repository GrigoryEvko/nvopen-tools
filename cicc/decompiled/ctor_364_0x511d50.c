// Function: ctor_364
// Address: 0x511d50
//
int ctor_364()
{
  int v1; // [rsp+Ch] [rbp-34h] BYREF
  const char *v2; // [rsp+10h] [rbp-30h] BYREF
  __int64 v3; // [rsp+18h] [rbp-28h]

  v2 = "Force to use handle for textures";
  v3 = 32;
  v1 = 1;
  sub_21DC420(&unk_4FD4060, "force-texture-handle", &v1, &v2);
  __cxa_atexit(sub_12EDEC0, &unk_4FD4060, &qword_4A427C0);
  v2 = "Force to use handle for surfaces";
  v3 = 32;
  v1 = 1;
  sub_21DC420(&unk_4FD3F80, "force-surface-handle", &v1, &v2);
  __cxa_atexit(sub_12EDEC0, &unk_4FD3F80, &qword_4A427C0);
  v2 = "Force to use handle for samplers";
  v3 = 32;
  v1 = 1;
  sub_21DC420(&unk_4FD3EA0, "force-sampler-handle", &v1, &v2);
  return __cxa_atexit(sub_12EDEC0, &unk_4FD3EA0, &qword_4A427C0);
}
