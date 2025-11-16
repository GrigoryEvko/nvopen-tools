// Function: sub_26F6BC0
// Address: 0x26f6bc0
//
__int64 __fastcall sub_26F6BC0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // r13
  unsigned __int8 **v7; // rax
  __int64 v8; // r12
  unsigned __int8 *v9; // r14
  __int64 result; // rax
  __int64 v11; // [rsp+8h] [rbp-98h]
  __int64 v12; // [rsp+8h] [rbp-98h]
  __int64 v13; // [rsp+8h] [rbp-98h]
  __int64 v14; // [rsp+8h] [rbp-98h]
  __int64 v15; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v16; // [rsp+20h] [rbp-80h]
  unsigned int v17; // [rsp+28h] [rbp-78h]
  unsigned __int64 v18; // [rsp+30h] [rbp-70h]
  unsigned int v19; // [rsp+38h] [rbp-68h]
  char v20; // [rsp+40h] [rbp-60h]
  unsigned __int64 v21; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+58h] [rbp-48h]
  unsigned __int64 v23; // [rsp+60h] [rbp-40h]
  unsigned int v24; // [rsp+68h] [rbp-38h]
  char v25; // [rsp+70h] [rbp-30h]

  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a1 + 80);
  v20 = 0;
  v6 = sub_ACD640(v5, v4, 0);
  v7 = *(unsigned __int8 ***)a2;
  v8 = *(_QWORD *)(a1 + 56);
  v9 = *v7;
  v25 = 0;
  v15 = v6;
  result = sub_AD9FD0(v8, v9, &v15, 1, 0, (__int64)&v21, 0);
  if ( v25 )
  {
    v25 = 0;
    if ( v24 > 0x40 && v23 )
    {
      v13 = result;
      j_j___libc_free_0_0(v23);
      result = v13;
    }
    if ( v22 > 0x40 && v21 )
    {
      v14 = result;
      j_j___libc_free_0_0(v21);
      result = v14;
    }
  }
  if ( v20 )
  {
    v20 = 0;
    if ( v19 > 0x40 && v18 )
    {
      v11 = result;
      j_j___libc_free_0_0(v18);
      result = v11;
    }
    if ( v17 > 0x40 )
    {
      if ( v16 )
      {
        v12 = result;
        j_j___libc_free_0_0(v16);
        return v12;
      }
    }
  }
  return result;
}
