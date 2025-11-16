// Function: sub_26BE300
// Address: 0x26be300
//
bool __fastcall sub_26BE300(_QWORD *a1, _QWORD *a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rax
  bool v4; // cf
  bool v5; // zf
  int *v7; // r13
  size_t v8; // r12
  int *v9; // r14
  size_t v10; // r13
  _QWORD v11[2]; // [rsp+0h] [rbp-E0h] BYREF
  int v12[52]; // [rsp+10h] [rbp-D0h] BYREF

  v2 = sub_EF9210(a1);
  v3 = sub_EF9210(a2);
  v4 = v2 < v3;
  v5 = v2 == v3;
  if ( v2 == v3 )
  {
    v7 = (int *)a1[2];
    v8 = a1[3];
    if ( v7 )
    {
      sub_C7D030(v12);
      sub_C7D280(v12, v7, v8);
      sub_C7D290(v12, v11);
      v8 = v11[0];
    }
    v9 = (int *)a2[2];
    v10 = a2[3];
    if ( v9 )
    {
      sub_C7D030(v12);
      sub_C7D280(v12, v9, v10);
      sub_C7D290(v12, v11);
      v10 = v11[0];
    }
    v4 = v10 < v8;
    v5 = v10 == v8;
  }
  return !v4 && !v5;
}
