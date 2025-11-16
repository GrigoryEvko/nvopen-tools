// Function: sub_2D85950
// Address: 0x2d85950
//
char __fastcall sub_2D85950(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // r14
  char v6; // bl
  unsigned __int64 v8[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v9; // [rsp+10h] [rbp-30h]

  v2 = a1[2];
  v8[0] = 0;
  v8[1] = 0;
  v9 = v2;
  if ( v2 == -4096 || v2 == 0 || v2 == -8192 )
  {
    v3 = a2[2];
    if ( v2 == v3 )
      return v2;
    goto LABEL_8;
  }
  sub_BD6050(v8, *a1 & 0xFFFFFFFFFFFFFFF8LL);
  v3 = a2[2];
  v4 = a1[2];
  if ( v3 != v4 )
  {
    if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
      sub_BD60C0(a1);
LABEL_8:
    a1[2] = v3;
    if ( v3 != -4096 && v3 != 0 && v3 != -8192 )
      sub_BD73F0((__int64)a1);
    v5 = v9;
    v3 = a2[2];
    goto LABEL_12;
  }
  v5 = v9;
LABEL_12:
  LOBYTE(v2) = v5 != -8192;
  v6 = v2 & (v5 != 0 && v5 != -4096);
  if ( v5 != v3 )
  {
    if ( v3 != 0 && v3 != -4096 && v3 != -8192 )
      sub_BD60C0(a2);
    a2[2] = v5;
    if ( v6 )
      sub_BD73F0((__int64)a2);
    LOBYTE(v2) = v9 != -8192;
    v6 = v2 & (v9 != 0 && v9 != -4096);
  }
  if ( v6 )
    LOBYTE(v2) = sub_BD60C0(v8);
  return v2;
}
