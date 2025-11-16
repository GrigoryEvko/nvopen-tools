// Function: sub_3111B50
// Address: 0x3111b50
//
void __fastcall sub_3111B50(int *a1, __int64 a2)
{
  __int64 *(__fastcall *v2)(__int64 *, __int64); // rax
  unsigned __int8 *v3[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v4; // [rsp+10h] [rbp-20h] BYREF

  v2 = *(__int64 *(__fastcall **)(__int64 *, __int64))(*(_QWORD *)a1 + 24LL);
  if ( v2 == sub_3111B20 )
    sub_3111610((__int64 *)v3, a1[2], (__int64)(a1 + 4));
  else
    v2((__int64 *)v3, (__int64)a1);
  sub_CB6200(a2, v3[0], (size_t)v3[1]);
  if ( (__int64 *)v3[0] != &v4 )
    j_j___libc_free_0((unsigned __int64)v3[0]);
}
