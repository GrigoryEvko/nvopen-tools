// Function: sub_9C39B0
// Address: 0x9c39b0
//
_QWORD *__fastcall sub_9C39B0(_QWORD *a1, const void *a2, __int64 a3)
{
  size_t v3; // r13
  char *v4; // rbx
  char *v5; // rax

  v3 = 8 * a3;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( (unsigned __int64)(8 * a3) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v4 = 0;
  if ( v3 )
  {
    v5 = (char *)sub_22077B0(8 * a3);
    v4 = &v5[v3];
    *a1 = v5;
    a1[2] = &v5[v3];
    memcpy(v5, a2, v3);
  }
  a1[1] = v4;
  return a1;
}
