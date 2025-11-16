// Function: sub_21D79A0
// Address: 0x21d79a0
//
__int64 __fastcall sub_21D79A0(__int64 a1, char *a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned int v4; // r12d
  _QWORD *v6; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v7; // [rsp+8h] [rbp-58h]
  _QWORD v8[2]; // [rsp+10h] [rbp-50h] BYREF
  char *nptr; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v10[6]; // [rsp+30h] [rbp-30h] BYREF

  v2 = -1;
  v6 = v8;
  if ( a2 )
    v2 = (__int64)&a2[strlen(a2)];
  sub_21CA7A0((__int64 *)&v6, a2, v2);
  v3 = sub_2241950(&v6, "_param_", -1, 7) + 1;
  if ( v3 > v7 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
  nptr = (char *)v10;
  sub_21CA7A0((__int64 *)&nptr, (_BYTE *)v6 + v3, (__int64)v6 + v7);
  v4 = strtol(nptr, 0, 10);
  if ( nptr != (char *)v10 )
    j_j___libc_free_0(nptr, v10[0] + 1LL);
  if ( v6 != v8 )
    j_j___libc_free_0(v6, v8[0] + 1LL);
  return v4;
}
