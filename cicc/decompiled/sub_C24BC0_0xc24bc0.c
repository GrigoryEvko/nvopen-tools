// Function: sub_C24BC0
// Address: 0xc24bc0
//
__int64 __fastcall sub_C24BC0(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  int v9; // eax
  __int64 v10; // rdx
  _QWORD v11[2]; // [rsp+0h] [rbp-70h] BYREF
  char v12; // [rsp+10h] [rbp-60h]
  _QWORD v13[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v14; // [rsp+40h] [rbp-30h]

  v13[0] = a2;
  v14 = 261;
  v13[1] = a3;
  sub_C1F520((__int64)v11, (__int64)v13, a4);
  if ( (v12 & 1) != 0 && (v9 = v11[0], v10 = v11[1], LODWORD(v11[0])) )
  {
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v9;
    *(_QWORD *)(a1 + 8) = v10;
  }
  else
  {
    sub_C247D0(a1, v11, a5, a6);
    if ( (v12 & 1) == 0 && v11[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
  }
  return a1;
}
