// Function: sub_1ACAA60
// Address: 0x1acaa60
//
__int64 __fastcall sub_1ACAA60(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 *v4; // r13
  unsigned __int16 *v5; // r14
  unsigned int v6; // r15d
  unsigned int v7; // eax
  __int64 result; // rax
  __int16 v9; // r15
  __int16 v10; // ax
  __int16 v11; // r15
  __int16 v12; // ax
  unsigned int v13; // r15d
  unsigned int v14; // eax
  void *v15; // r14
  __int64 *v16; // rsi
  __int64 *v17; // rsi
  unsigned int v19; // [rsp+8h] [rbp-58h]
  unsigned int v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-38h]

  v4 = *(unsigned __int16 **)(a3 + 8);
  v5 = *(unsigned __int16 **)(a2 + 8);
  v6 = sub_16982D0((__int64)v4);
  v7 = sub_16982D0((__int64)v5);
  result = sub_1ACA9E0(a1, v7, v6);
  if ( !(_DWORD)result )
  {
    v9 = sub_16982E0(v4);
    v10 = sub_16982E0(v5);
    result = sub_1ACA9E0(a1, v10, v9);
    if ( !(_DWORD)result )
    {
      v11 = sub_16982F0((__int64)v4);
      v12 = sub_16982F0((__int64)v5);
      result = sub_1ACA9E0(a1, v12, v11);
      if ( !(_DWORD)result )
      {
        v13 = sub_1698300((__int64)v4);
        v14 = sub_1698300((__int64)v5);
        result = sub_1ACA9E0(a1, v14, v13);
        if ( !(_DWORD)result )
        {
          v15 = sub_16982C0();
          v16 = (__int64 *)(a3 + 8);
          if ( *(void **)(a3 + 8) == v15 )
            sub_169D930((__int64)&v23, (__int64)v16);
          else
            sub_169D7E0((__int64)&v23, v16);
          v17 = (__int64 *)(a2 + 8);
          if ( *(void **)(a2 + 8) == v15 )
            sub_169D930((__int64)&v21, (__int64)v17);
          else
            sub_169D7E0((__int64)&v21, v17);
          result = sub_1ACAA10(a1, (__int64)&v21, (__int64)&v23);
          if ( v22 > 0x40 && v21 )
          {
            v19 = result;
            j_j___libc_free_0_0(v21);
            result = v19;
          }
          if ( v24 > 0x40 )
          {
            if ( v23 )
            {
              v20 = result;
              j_j___libc_free_0_0(v23);
              return v20;
            }
          }
        }
      }
    }
  }
  return result;
}
