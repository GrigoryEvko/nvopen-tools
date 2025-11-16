// Function: sub_22CF6C0
// Address: 0x22cf6c0
//
__int64 __fastcall sub_22CF6C0(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  __int64 result; // rax
  __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+8h] [rbp-68h]
  _BYTE v17[8]; // [rsp+10h] [rbp-60h] BYREF
  _BYTE *v18; // [rsp+18h] [rbp-58h]
  unsigned int v19; // [rsp+20h] [rbp-50h]
  unsigned __int64 v20; // [rsp+28h] [rbp-48h]
  unsigned int v21; // [rsp+30h] [rbp-40h]

  v10 = sub_AA4B30(a5);
  v11 = sub_22C1480(a1, v10);
  sub_22CF010((__int64)v17, v11, a3, a5, a6, a7);
  v12 = v10 + 312;
  if ( v17[0] == 2 )
  {
    result = sub_9719A0(a2, v18, a4, v12, 0, 0);
    if ( (unsigned int)v17[0] - 4 > 1 )
      return result;
  }
  else
  {
    result = sub_22BE9F0(a2, a4, (__int64)v17, v12);
    if ( (unsigned int)v17[0] - 4 > 1 )
      return result;
  }
  if ( v21 > 0x40 && v20 )
  {
    v15 = result;
    j_j___libc_free_0_0(v20);
    result = v15;
  }
  if ( v19 > 0x40 )
  {
    if ( v18 )
    {
      v16 = result;
      j_j___libc_free_0_0((unsigned __int64)v18);
      return v16;
    }
  }
  return result;
}
