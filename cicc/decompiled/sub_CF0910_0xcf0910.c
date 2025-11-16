// Function: sub_CF0910
// Address: 0xcf0910
//
void __fastcall sub_CF0910(unsigned __int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi
  unsigned __int64 v3; // [rsp+0h] [rbp-40h] BYREF
  int v4; // [rsp+8h] [rbp-38h] BYREF
  __int64 v5; // [rsp+10h] [rbp-30h]
  int *v6; // [rsp+18h] [rbp-28h]
  int *v7; // [rsp+20h] [rbp-20h]
  __int64 v8; // [rsp+28h] [rbp-18h]

  v4 = 0;
  v5 = 0;
  v6 = &v4;
  v7 = &v4;
  v8 = 0;
  sub_CF0810(a1, &v3);
  v1 = v5;
  while ( v1 )
  {
    sub_CEF340(*(_QWORD *)(v1 + 24));
    v2 = v1;
    v1 = *(_QWORD *)(v1 + 16);
    j_j___libc_free_0(v2, 40);
  }
}
