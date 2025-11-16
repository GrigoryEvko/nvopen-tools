// Function: sub_101D570
// Address: 0x101d570
//
__int64 __fastcall sub_101D570(unsigned int a1, __int64 a2, _BYTE *a3, char a4, __m128i *a5, int a6)
{
  unsigned __int8 *v9; // r12
  __int64 v11; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-48h]
  _BYTE *v13; // [rsp+10h] [rbp-40h]
  unsigned int v14; // [rsp+18h] [rbp-38h]

  v9 = sub_101CD30(a1, (_BYTE *)a2, a3, 0, a5, a6);
  if ( v9 )
    return (__int64)v9;
  if ( (_BYTE *)a2 != a3 )
  {
    if ( (unsigned __int8)sub_1003090((__int64)a5, (unsigned __int8 *)a2) )
    {
      v9 = (unsigned __int8 *)a2;
      if ( !a4 )
        return sub_AD6530(*(_QWORD *)(a2 + 8), a2);
      return (__int64)v9;
    }
    if ( !a4 )
      return (__int64)v9;
    sub_9AC330((__int64)&v11, a2, 0, a5);
    if ( v14 <= 0x40 )
    {
      if ( ((unsigned __int8)v13 & 1) != 0 )
        goto LABEL_11;
    }
    else
    {
      if ( (*v13 & 1) != 0 )
      {
        j_j___libc_free_0_0(v13);
LABEL_11:
        if ( v12 > 0x40 )
        {
          if ( v11 )
            j_j___libc_free_0_0(v11);
        }
        return a2;
      }
      j_j___libc_free_0_0(v13);
    }
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
    return (__int64)v9;
  }
  return sub_AD6530(*(_QWORD *)(a2 + 8), a2);
}
