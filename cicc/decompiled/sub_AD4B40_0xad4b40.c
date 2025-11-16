// Function: sub_AD4B40
// Address: 0xad4b40
//
__int64 __fastcall sub_AD4B40(unsigned int a1, unsigned __int64 a2, __int64 **a3, char a4)
{
  unsigned __int8 v4; // r13
  __int64 result; // rax
  __int64 v8; // rdi
  __int64 v9; // [rsp+0h] [rbp-90h]
  __int64 v10; // [rsp+0h] [rbp-90h]
  unsigned __int64 v11; // [rsp+8h] [rbp-88h] BYREF
  __int16 v12; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 *v13; // [rsp+18h] [rbp-78h]
  __int64 v14; // [rsp+20h] [rbp-70h]
  __int64 v15; // [rsp+28h] [rbp-68h]
  __int64 v16; // [rsp+30h] [rbp-60h]
  __int64 v17; // [rsp+38h] [rbp-58h]
  __int64 v18; // [rsp+40h] [rbp-50h]
  unsigned int v19; // [rsp+48h] [rbp-48h]
  __int64 v20; // [rsp+50h] [rbp-40h]
  unsigned int v21; // [rsp+58h] [rbp-38h]
  char v22; // [rsp+60h] [rbp-30h]

  v4 = a1;
  v11 = a2;
  result = sub_AA93C0(a1, a2, (__int64)a3);
  if ( !result && !a4 )
  {
    v8 = **a3;
    v12 = v4;
    v13 = &v11;
    v14 = 1;
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v22 = 0;
    result = sub_AD4210(v8 + 2120, (__int64)a3, &v12);
    if ( v22 )
    {
      v22 = 0;
      if ( v21 > 0x40 && v20 )
      {
        v9 = result;
        j_j___libc_free_0_0(v20);
        result = v9;
      }
      if ( v19 > 0x40 )
      {
        if ( v18 )
        {
          v10 = result;
          j_j___libc_free_0_0(v18);
          return v10;
        }
      }
    }
  }
  return result;
}
