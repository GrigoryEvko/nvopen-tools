// Function: sub_BC6270
// Address: 0xbc6270
//
bool __fastcall sub_BC6270(_BYTE *a1, __int64 a2)
{
  bool result; // al
  bool v3; // [rsp+Fh] [rbp-51h]
  const void *v4[2]; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v5[6]; // [rsp+30h] [rbp-30h] BYREF

  if ( byte_4F82818 || !(unsigned int)sub_2207590(&byte_4F82818) )
  {
    result = 1;
    if ( !qword_4F82838 )
      return result;
  }
  else
  {
    sub_BC5EA0(qword_4F82820, (char *)qword_4F829E8, (char *)qword_4F829F0, 0);
    __cxa_atexit((void (*)(void *))sub_8565C0, qword_4F82820, &qword_4A427C0);
    sub_2207640(&byte_4F82818);
    result = 1;
    if ( !qword_4F82838 )
      return result;
  }
  v4[0] = v5;
  sub_BC5450((__int64 *)v4, a1, (__int64)&a1[a2]);
  result = sub_BB97F0(qword_4F82820, v4) != 0;
  if ( v4[0] != v5 )
  {
    v3 = result;
    j_j___libc_free_0(v4[0], v5[0] + 1LL);
    return v3;
  }
  return result;
}
