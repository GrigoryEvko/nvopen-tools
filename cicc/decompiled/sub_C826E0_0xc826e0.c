// Function: sub_C826E0
// Address: 0xc826e0
//
__int64 __fastcall sub_C826E0(__int64 a1, __int64 a2, char a3)
{
  char *v4; // rdi
  unsigned int v5; // eax
  __int64 v6; // rcx
  __int64 v7; // r8
  unsigned int v8; // ebx
  struct stat stat_buf; // [rsp+0h] [rbp-150h] BYREF
  _QWORD v11[3]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE v12[168]; // [rsp+A8h] [rbp-A8h] BYREF

  v11[0] = v12;
  v11[1] = 0;
  v11[2] = 128;
  v4 = (char *)sub_CA12A0(a1, v11);
  if ( a3 )
    v5 = sub_39FAD60(v4, &stat_buf);
  else
    v5 = sub_39FAD70(v4, &stat_buf);
  v8 = sub_C7FD70(v5, (__int64 *)&stat_buf, a2, v6, v7);
  if ( (_BYTE *)v11[0] != v12 )
    _libc_free(v11[0], &stat_buf);
  return v8;
}
