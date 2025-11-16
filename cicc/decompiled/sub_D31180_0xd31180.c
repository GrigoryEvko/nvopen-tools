// Function: sub_D31180
// Address: 0xd31180
//
__int64 __fastcall sub_D31180(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 result; // rax
  unsigned __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int8 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  __int64 v21; // [rsp+18h] [rbp-38h]

  v13 = sub_9208B0(a4, a2);
  v15 = v14;
  v20 = v13;
  v16 = v13;
  result = 0;
  v21 = v15;
  if ( !(_BYTE)v15 )
  {
    v18 = (unsigned __int64)(v16 + 7) >> 3;
    LODWORD(v21) = sub_AE43F0(a4, *(_QWORD *)(a1 + 8));
    if ( (unsigned int)v21 > 0x40 )
      sub_C43690((__int64)&v20, v18, 0);
    else
      v20 = v18;
    result = sub_D30F00((unsigned __int8 *)a1, a3, (__int64)&v20, a4, a5, a6, a7, a8);
    if ( (unsigned int)v21 > 0x40 )
    {
      if ( v20 )
      {
        v19 = result;
        j_j___libc_free_0_0(v20);
        return v19;
      }
    }
  }
  return result;
}
