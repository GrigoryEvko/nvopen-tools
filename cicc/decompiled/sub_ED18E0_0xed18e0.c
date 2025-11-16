// Function: sub_ED18E0
// Address: 0xed18e0
//
__int64 __fastcall sub_ED18E0(__int64 a1, size_t a2)
{
  __int64 v2; // rax
  size_t v3; // rdi
  __int64 v4; // r14
  void *v5; // rax
  __int64 v7; // [rsp+0h] [rbp-50h] BYREF
  size_t v8; // [rsp+8h] [rbp-48h]
  _WORD *v9; // [rsp+10h] [rbp-40h] BYREF
  size_t v10; // [rsp+18h] [rbp-38h]
  _QWORD v11[6]; // [rsp+20h] [rbp-30h] BYREF

  v7 = a1;
  v8 = a2;
  v9 = v11;
  sub_ED0450((__int64 *)&v9, ".__uniq.", (__int64)"");
  v2 = sub_C931B0(&v7, v9, v10, 0);
  v3 = 0;
  if ( v2 != -1 )
    v3 = v10 + v2;
  if ( v3 >= v8
    || (v4 = v7, (v5 = memchr((const void *)(v7 + v3), 46, v8 - v3)) == 0)
    || (unsigned __int64)v5 - v4 - 1 > 0xFFFFFFFFFFFFFFFDLL )
  {
    v4 = v7;
  }
  if ( v9 != (_WORD *)v11 )
    j_j___libc_free_0(v9, v11[0] + 1LL);
  return v4;
}
