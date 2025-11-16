// Function: sub_26E20C0
// Address: 0x26e20c0
//
bool __fastcall sub_26E20C0(__int64 a1, __int64 a2, __int64 *a3)
{
  int *v4; // r15
  size_t v5; // r14
  _QWORD *v6; // rax
  __int64 v7; // rax
  size_t v9[2]; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v10[26]; // [rsp+10h] [rbp-D0h] BYREF

  *a3 = 0;
  v4 = *(int **)a2;
  v5 = *(_QWORD *)(a2 + 8);
  if ( *(_QWORD *)a2 )
  {
    sub_C7D030(v10);
    sub_C7D280((int *)v10, v4, v5);
    sub_C7D290(v10, v9);
    v5 = v9[0];
  }
  v10[0] = v5;
  v6 = sub_26C56D0((_QWORD *)(a1 + 272), v10);
  if ( v6 )
  {
    v7 = v6[2];
    *a3 = v7;
  }
  else
  {
    v7 = *a3;
  }
  return v7 == 0;
}
