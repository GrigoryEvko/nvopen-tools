// Function: sub_38A29E0
// Address: 0x38a29e0
//
__int64 __fastcall sub_38A29E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  __int64 v8; // rdi
  bool v9; // zf
  unsigned __int64 v10; // rsi
  int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // [rsp+0h] [rbp-60h] BYREF
  __int64 v17; // [rsp+8h] [rbp-58h]
  _QWORD v18[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v19; // [rsp+20h] [rbp-40h]
  _QWORD v20[2]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v21; // [rsp+40h] [rbp-20h]

  v8 = a1 + 8;
  v9 = *(_BYTE *)(a4 + 8) == 0;
  v16 = a2;
  v17 = a3;
  if ( v9 )
  {
    v13 = sub_3887100(v8);
    v14 = v16;
    v15 = v17;
    *(_DWORD *)(a1 + 64) = v13;
    return sub_38A2910(a1, v14, v15, a4, a5, a6, a7);
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 56);
    v19 = 1283;
    v18[0] = "field '";
    v18[1] = &v16;
    v20[0] = v18;
    v21 = 770;
    v20[1] = "' cannot be specified more than once";
    return sub_38814C0(v8, v10, (__int64)v20);
  }
}
