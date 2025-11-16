// Function: ctor_148
// Address: 0x4cdcc0
//
int ctor_148()
{
  _QWORD v1[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v2; // [rsp+10h] [rbp-20h]

  v1[0] = 0;
  v1[1] = 0;
  v2 = 0;
  sub_1633F60(&unk_4F9ED00, v1);
  if ( v1[0] )
    j_j___libc_free_0(v1[0], v2 - v1[0]);
  return __cxa_atexit(sub_142C5C0, &unk_4F9ED00, &qword_4A427C0);
}
