// Function: sub_CE8980
// Address: 0xce8980
//
__int64 __fastcall sub_CE8980(unsigned __int8 *a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned __int8 *v3; // rax
  _QWORD v5[2]; // [rsp+0h] [rbp-40h] BYREF
  char v6[48]; // [rsp+10h] [rbp-30h] BYREF

  v2 = 0;
  v5[0] = v6;
  strcpy(v6, "wroimage");
  v5[1] = 8;
  v3 = sub_BD3990(a1, a2);
  if ( *v3 == 22 )
    v2 = sub_CE7A30((__int64)v3, (__int64)v5);
  if ( (char *)v5[0] != v6 )
    j_j___libc_free_0(v5[0], *(_QWORD *)v6 + 1LL);
  return v2;
}
