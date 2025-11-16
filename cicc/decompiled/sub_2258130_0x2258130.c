// Function: sub_2258130
// Address: 0x2258130
//
void __fastcall sub_2258130(__int64 a1, char *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rdx
  char *v4; // rcx
  __int64 v5[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v6[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = -1;
  v5[0] = (__int64)v6;
  if ( a2 )
    v2 = (__int64)&a2[strlen(a2)];
  sub_22579E0(v5, a2, v2);
  sub_CEB520(v5, (__int64)a2, v3, v4);
  if ( (_QWORD *)v5[0] != v6 )
    j_j___libc_free_0(v5[0]);
}
