// Function: sub_32237B0
// Address: 0x32237b0
//
__int64 __fastcall sub_32237B0(_QWORD *a1, __int64 a2)
{
  const void *v2; // r15
  __int64 v3; // r13
  unsigned __int64 v4; // r13
  __int64 v5; // r12
  char *v6; // rbx
  char *v7; // rax
  char *v8; // rax
  char *v10; // [rsp+0h] [rbp-50h] BYREF
  char *v11; // [rsp+8h] [rbp-48h]
  char *v12; // [rsp+10h] [rbp-40h]

  v2 = *(const void **)(a2 + 16);
  v3 = *(_QWORD *)(a2 + 24);
  v10 = 0;
  v11 = 0;
  v4 = v3 - (_QWORD)v2;
  v12 = 0;
  if ( v4 > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v5 = (__int64)a1;
  v6 = 0;
  if ( v4 )
  {
    v7 = (char *)sub_22077B0(v4);
    v6 = &v7[v4];
    v10 = v7;
    v12 = &v7[v4];
    memcpy(v7, v2, v4);
  }
  v11 = v6;
  if ( (unsigned __int8)sub_AF4460((__int64)a1) && (unsigned __int8)sub_AF4460(a2) )
    sub_3223650(&v10, 159);
  v8 = v10;
  if ( v11 != v10 )
  {
    v5 = sub_B0DED0(a1, v10, (v11 - v10) >> 3);
    v8 = v10;
  }
  if ( v8 )
    j_j___libc_free_0((unsigned __int64)v8);
  return v5;
}
