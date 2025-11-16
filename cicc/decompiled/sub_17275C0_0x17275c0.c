// Function: sub_17275C0
// Address: 0x17275c0
//
__int64 __fastcall sub_17275C0(__int64 a1, _BYTE *a2, _DWORD *a3, __int64 **a4, __int64 *a5, __int64 *a6)
{
  unsigned int v9; // r12d
  unsigned __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-38h]

  v12 = 1;
  v11 = 0;
  v9 = sub_14CF800(a1, a2, a3, a4, &v11, 1u);
  if ( (_BYTE)v9 )
  {
    *a5 = sub_15A1070(**a4, (__int64)&v11);
    *a6 = sub_15A0680(**a4, 0, 0);
  }
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  return v9;
}
