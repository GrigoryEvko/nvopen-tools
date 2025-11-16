// Function: sub_1C3F040
// Address: 0x1c3f040
//
struct __jmp_buf_tag *__fastcall sub_1C3F040(__int64 a1)
{
  unsigned __int8 *v1; // rax
  struct __jmp_buf_tag *result; // rax
  unsigned __int8 *v3; // rax
  __int64 v4[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v5[4]; // [rsp+10h] [rbp-20h] BYREF

  v1 = (unsigned __int8 *)sub_16D40F0((__int64)qword_4FBB4D0);
  if ( v1 )
    result = (struct __jmp_buf_tag *)*v1;
  else
    result = (struct __jmp_buf_tag *)LOBYTE(qword_4FBB4D0[2]);
  if ( !(_BYTE)result )
  {
    LOWORD(v4[0]) = 257;
    sub_1C3EA60(a1, (char *)v4, 1);
    v3 = (unsigned __int8 *)sub_16D40F0((__int64)qword_4FBB350);
    if ( v3 )
      result = (struct __jmp_buf_tag *)*v3;
    else
      result = (struct __jmp_buf_tag *)LOBYTE(qword_4FBB350[2]);
    if ( (_BYTE)result )
    {
      v4[0] = (__int64)v5;
      sub_CEB5A0(v4, "warning treated as error.", (__int64)"");
      result = sub_1C3EF50((__int64)v4);
      if ( (_QWORD *)v4[0] != v5 )
        return (struct __jmp_buf_tag *)j_j___libc_free_0(v4[0], v5[0] + 1LL);
    }
  }
  return result;
}
