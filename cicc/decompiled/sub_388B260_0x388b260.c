// Function: sub_388B260
// Address: 0x388b260
//
__int64 __fastcall sub_388B260(__int64 **a1, __int64 *a2)
{
  unsigned int v2; // r12d
  _QWORD *v4; // [rsp+0h] [rbp-40h] BYREF
  size_t v5; // [rsp+8h] [rbp-38h]
  _BYTE v6[48]; // [rsp+10h] [rbp-30h] BYREF

  v4 = v6;
  v5 = 0;
  v6[0] = 0;
  v2 = sub_388B0A0((__int64)a1, (unsigned __int64 *)&v4);
  if ( !(_BYTE)v2 )
    *a2 = sub_161FF10(*a1, v4, v5);
  if ( v4 != (_QWORD *)v6 )
    j_j___libc_free_0((unsigned __int64)v4);
  return v2;
}
