// Function: sub_2F57B40
// Address: 0x2f57b40
//
__int64 __fastcall sub_2F57B40(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  unsigned int v7; // eax
  __int64 v8; // r9
  char v10; // cl
  unsigned int v11; // [rsp+4h] [rbp-3Ch] BYREF
  _QWORD v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v11 = 0;
  v12[0] = 0;
  v6 = sub_2F525E0((__int64)a1);
  if ( *(_DWORD *)(a1[124] + 696LL) && sub_2F524F0(a1, a1[3022]) )
  {
    v11 = 1;
    v12[0] = -1;
    v7 = sub_2F579F0((__int64)a1, a2, a3, (__int64)v12, &v11, 0);
    v10 = 1;
  }
  else
  {
    v12[0] = v6;
    v7 = sub_2F579F0((__int64)a1, a2, a3, (__int64)v12, &v11, 0);
    if ( v7 == -1 )
      return 0;
    v10 = 0;
  }
  return sub_2F53540((__int64)a1, a2, v7, v10, a4, v8);
}
