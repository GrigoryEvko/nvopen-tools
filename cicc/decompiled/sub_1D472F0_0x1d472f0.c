// Function: sub_1D472F0
// Address: 0x1d472f0
//
unsigned __int64 __fastcall sub_1D472F0(__int64 a1, _QWORD *a2, __int64 a3, char a4)
{
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD v12[2]; // [rsp+0h] [rbp-B0h] BYREF
  _QWORD v13[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v14; // [rsp+20h] [rbp-90h]
  char *v15; // [rsp+30h] [rbp-80h]
  char v16; // [rsp+40h] [rbp-70h]
  char v17; // [rsp+41h] [rbp-6Fh]
  _QWORD v18[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v19; // [rsp+60h] [rbp-50h]
  _BYTE *v20[2]; // [rsp+70h] [rbp-40h] BYREF
  __int64 v21; // [rsp+80h] [rbp-30h] BYREF

  if ( !*(_QWORD *)(a3 + 32) || a4 )
  {
    v17 = 1;
    v15 = ")";
    v16 = 3;
    v12[0] = sub_1E0A440();
    v13[0] = " (in function: ";
    v13[1] = v12;
    v12[1] = v6;
    LOWORD(v14) = 1283;
    v18[1] = ")";
    v18[0] = v13;
    LOWORD(v19) = 770;
    sub_16E2FC0((__int64 *)v20, (__int64)v18);
    sub_15CAB20(a3, v20[0], (size_t)v20[1]);
    if ( (__int64 *)v20[0] != &v21 )
      j_j___libc_free_0(v20[0], v21 + 1);
    if ( a4 )
    {
      sub_15CA8E0((__int64 *)v20, a3, v7, v8, v9, v10);
      sub_16BD160((__int64)v20, 1u);
    }
  }
  return sub_143AA50(a2, a3);
}
