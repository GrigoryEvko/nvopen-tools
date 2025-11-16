// Function: sub_3117DB0
// Address: 0x3117db0
//
void __fastcall sub_3117DB0(__int64 *a1, char *a2)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  char v6[8]; // [rsp+0h] [rbp-50h] BYREF
  int v7; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 v8; // [rsp+10h] [rbp-40h]
  int *v9; // [rsp+18h] [rbp-38h]
  int *v10; // [rsp+20h] [rbp-30h]
  __int64 v11; // [rsp+28h] [rbp-28h]

  v2 = (__int64)a2;
  v7 = 0;
  v8 = 0;
  v9 = &v7;
  v10 = &v7;
  v11 = 0;
  if ( (unsigned __int8)sub_CB4D10((__int64)a2, (__int64)a2) )
  {
    a2 = v6;
    sub_3117B40(v2, (__int64)v6);
  }
  sub_CB0D90(v2, (__int64)a2);
  sub_3116A00(a1, (__int64)v6);
  v3 = v8;
  while ( v3 )
  {
    v4 = v3;
    sub_31152F0(*(_QWORD **)(v3 + 24));
    v5 = *(_QWORD *)(v3 + 56);
    v3 = *(_QWORD *)(v3 + 16);
    if ( v5 )
      j_j___libc_free_0(v5);
    j_j___libc_free_0(v4);
  }
}
