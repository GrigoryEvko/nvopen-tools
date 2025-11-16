// Function: sub_58F9A0
// Address: 0x58f9a0
//
__int64 __fastcall sub_58F9A0(_QWORD *a1, const void *a2, __int64 a3)
{
  __int64 result; // rax
  size_t v4; // r12
  char *v5; // r14
  char *v6; // rax

  result = 0x7FFFFFFFFFFFFFF8LL;
  v4 = 8 * a3;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( (unsigned __int64)(8 * a3) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v5 = 0;
  if ( v4 )
  {
    v6 = (char *)sub_22077B0(8 * a3);
    v5 = &v6[v4];
    *a1 = v6;
    a1[2] = &v6[v4];
    result = (__int64)memcpy(v6, a2, v4);
  }
  a1[1] = v5;
  return result;
}
