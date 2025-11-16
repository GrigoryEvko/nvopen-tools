// Function: sub_C136E0
// Address: 0xc136e0
//
__int64 __fastcall sub_C136E0(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, const char *, __int64, __int64),
        __int64 a3)
{
  _BYTE *v3; // rsi
  bool v4; // zf
  __int64 result; // rax
  __int64 (__fastcall *v6)(__int64, const char *, __int64, __int64); // [rsp+0h] [rbp-80h] BYREF
  __int64 v7; // [rsp+8h] [rbp-78h]
  __int64 v8; // [rsp+18h] [rbp-68h]
  __int64 v9[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v10[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v11; // [rsp+40h] [rbp-40h]
  __int64 v12; // [rsp+48h] [rbp-38h]
  __int64 v13; // [rsp+50h] [rbp-30h]

  v6 = a2;
  v7 = a3;
  v9[0] = (__int64)&v6;
  sub_C126E0((__int64 *)a1, (void (__fastcall *)(__int64, void **))sub_C11EA0, (__int64)v9);
  v3 = *(_BYTE **)(a1 + 232);
  v9[0] = (__int64)v10;
  sub_C11DF0(v9, v3, (__int64)&v3[*(_QWORD *)(a1 + 240)]);
  v4 = *(_DWORD *)(a1 + 284) == 3;
  result = *(unsigned int *)(a1 + 264);
  v11 = *(_QWORD *)(a1 + 264);
  v12 = *(_QWORD *)(a1 + 272);
  v13 = *(_QWORD *)(a1 + 280);
  if ( v4 )
  {
    result = (unsigned int)(result - 38);
    if ( (unsigned int)result <= 1 )
    {
      v8 = sub_BAA610(a1);
      if ( (_DWORD)v11 == 38
        || (result = (unsigned int)v8, BYTE4(v8)) && (result = (unsigned int)(v8 - 3), (unsigned int)result <= 1) )
      {
        result = v6(v7, "_GLOBAL_OFFSET_TABLE_", 21, 3);
      }
    }
  }
  if ( (_QWORD *)v9[0] != v10 )
    return j_j___libc_free_0(v9[0], v10[0] + 1LL);
  return result;
}
