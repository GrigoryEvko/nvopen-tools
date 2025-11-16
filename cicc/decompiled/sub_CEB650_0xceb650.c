// Function: sub_CEB650
// Address: 0xceb650
//
struct __jmp_buf_tag *__fastcall sub_CEB650(_QWORD *a1)
{
  unsigned __int8 *v1; // rax
  char *v2; // rcx
  struct __jmp_buf_tag *result; // rax
  unsigned __int8 *v4; // rax
  __int64 v5; // rdx
  char *v6; // rcx
  __int64 v7[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v8[4]; // [rsp+10h] [rbp-20h] BYREF

  v1 = (unsigned __int8 *)sub_C94E20((__int64)qword_4F86430);
  if ( v1 )
    result = (struct __jmp_buf_tag *)*v1;
  else
    result = (struct __jmp_buf_tag *)LOBYTE(qword_4F86430[2]);
  if ( !(_BYTE)result )
  {
    sub_CEB020(a1, 0x101u, 1, v2);
    v4 = (unsigned __int8 *)sub_C94E20((__int64)qword_4F862B0);
    if ( v4 )
      result = (struct __jmp_buf_tag *)*v4;
    else
      result = (struct __jmp_buf_tag *)LOBYTE(qword_4F862B0[2]);
    if ( (_BYTE)result )
    {
      v7[0] = (__int64)v8;
      sub_CEB5A0(v7, "warning treated as error.", (__int64)"");
      result = sub_CEB520(v7, (__int64)"warning treated as error.", v5, v6);
      if ( (_QWORD *)v7[0] != v8 )
        return (struct __jmp_buf_tag *)j_j___libc_free_0(v7[0], v8[0] + 1LL);
    }
  }
  return result;
}
