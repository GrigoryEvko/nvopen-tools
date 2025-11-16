// Function: sub_12B9F70
// Address: 0x12b9f70
//
char __fastcall sub_12B9F70(unsigned int a1)
{
  char *v1; // r12
  size_t v2; // rax
  size_t v3; // r14
  char *v4; // r8
  char *v5; // r15
  int v7; // eax
  char *v8; // [rsp+8h] [rbp-58h]
  char *name; // [rsp+10h] [rbp-50h] BYREF
  __int64 v10; // [rsp+18h] [rbp-48h]
  _QWORD v11[8]; // [rsp+20h] [rbp-40h] BYREF

  name = (char *)v11;
  sub_12B9B00((__int64 *)&name, (char)&unk_4281304 - 20, (_BYTE *)&unk_4281304 - 20, &unk_4281304);
  v1 = getenv(name);
  if ( name != (char *)v11 )
    j_j___libc_free_0(name, v11[0] + 1LL);
  if ( !v1 )
    return a1 > 0x63;
  v2 = strlen(v1);
  name = (char *)v11;
  v3 = v2;
  sub_12B9B00((__int64 *)&name, (char)&byte_42812E2[-6], &byte_42812E2[-6], byte_42812E2);
  v4 = name;
  if ( v3 == v10 )
  {
    if ( !v3 || (v8 = name, v7 = memcmp(v1, name, v3), v4 = v8, !v7) )
    {
      if ( v4 != (char *)v11 )
        j_j___libc_free_0(v4, v11[0] + 1LL);
      return 0;
    }
  }
  if ( v4 != (char *)v11 )
    j_j___libc_free_0(v4, v11[0] + 1LL);
  name = (char *)v11;
  sub_12B9B00((__int64 *)&name, (char)&byte_42812DB[-11], &byte_42812DB[-11], byte_42812DB);
  v5 = name;
  if ( v3 != v10 || v3 && memcmp(v1, name, v3) )
  {
    if ( v5 != (char *)v11 )
      j_j___libc_free_0(v5, v11[0] + 1LL);
    return a1 > 0x63;
  }
  if ( v5 != (char *)v11 )
    j_j___libc_free_0(v5, v11[0] + 1LL);
  return 1;
}
