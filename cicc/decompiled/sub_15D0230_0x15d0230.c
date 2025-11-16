// Function: sub_15D0230
// Address: 0x15d0230
//
char *__fastcall sub_15D0230(__int64 a1, char *a2, __int64 a3)
{
  char *v4; // r12
  __int64 v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 *v9; // rbx
  char *v10; // [rsp+8h] [rbp-48h] BYREF
  char *v11; // [rsp+10h] [rbp-40h] BYREF
  char *v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v10 = a2;
  v4 = (char *)sub_15CC960(a3, (__int64)a2);
  if ( !v4 )
  {
    v11 = a2;
    if ( (unsigned __int8)sub_15CE6E0(a1 + 24, (__int64 *)&v11, v12)
      && v12[0] != (char *)(*(_QWORD *)(a1 + 32) + 72LL * *(unsigned int *)(a1 + 48)) )
    {
      v4 = (char *)*((_QWORD *)v12[0] + 4);
    }
    v6 = sub_15D0230(a1, v4, a3);
    sub_15CC0B0((__int64 *)&v11, (__int64)v10, v6);
    v12[0] = v11;
    sub_15CE4A0(v6 + 24, v12);
    v4 = v11;
    v11 = 0;
    v7 = sub_15CFF10(a3 + 48, (__int64 *)&v10);
    v8 = v7[1];
    v9 = v7;
    v7[1] = (__int64)v4;
    if ( v8 )
    {
      sub_15CBC60(v8);
      v4 = (char *)v9[1];
    }
    if ( v11 )
      sub_15CBC60((__int64)v11);
  }
  return v4;
}
