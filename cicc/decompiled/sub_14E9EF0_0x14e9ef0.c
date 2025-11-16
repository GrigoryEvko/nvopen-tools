// Function: sub_14E9EF0
// Address: 0x14e9ef0
//
__int64 __fastcall sub_14E9EF0(_QWORD *a1, _BYTE *a2, _BYTE *a3)
{
  __int64 result; // rax
  size_t v4; // r13
  char *v7; // rdi

  result = 0x7FFFFFFFFFFFFFF8LL;
  v4 = a3 - a2;
  if ( (unsigned __int64)(a3 - a2) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v7 = 0;
  if ( v4 )
  {
    result = sub_22077B0(a3 - a2);
    v7 = (char *)result;
  }
  *a1 = v7;
  a1[2] = &v7[v4];
  if ( a3 != a2 )
    result = (__int64)memcpy(v7, a2, v4);
  a1[1] = &v7[v4];
  return result;
}
