// Function: ctor_697
// Address: 0x5a8210
//
int ctor_697()
{
  int v1; // [rsp+Ch] [rbp-34h] BYREF
  const char *v2; // [rsp+10h] [rbp-30h] BYREF
  __int64 v3; // [rsp+18h] [rbp-28h]

  v2 = "Force to use handle for textures";
  v3 = 32;
  v1 = 1;
  sub_36F7BC0(&unk_5041240, "force-texture-handle", &v1, &v2);
  __cxa_atexit(sub_984900, &unk_5041240, &qword_4A427C0);
  v2 = "Force to use handle for surfaces";
  v3 = 32;
  v1 = 1;
  sub_36F7BC0(&unk_5041160, "force-surface-handle", &v1, &v2);
  __cxa_atexit(sub_984900, &unk_5041160, &qword_4A427C0);
  v2 = "Force to use handle for samplers";
  v3 = 32;
  v1 = 1;
  sub_36F7BC0(&unk_5041080, "force-sampler-handle", &v1, &v2);
  return __cxa_atexit(sub_984900, &unk_5041080, &qword_4A427C0);
}
