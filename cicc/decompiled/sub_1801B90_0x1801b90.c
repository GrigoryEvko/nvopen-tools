// Function: sub_1801B90
// Address: 0x1801b90
//
__int64 __fastcall sub_1801B90(__int64 a1, _QWORD *a2, unsigned __int64 a3)
{
  unsigned int v4; // r12d
  _QWORD *v5; // rdx
  __int64 *v6; // r8
  __int64 v7; // rcx
  _QWORD *v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-28h]
  __int64 *v11; // [rsp+10h] [rbp-20h]
  unsigned int v12; // [rsp+18h] [rbp-18h]

  sub_140E6D0((__int64)&v9, a1, a2);
  if ( v10 <= 1 )
  {
    v4 = 0;
    if ( v12 <= 0x40 )
      return v4;
    v6 = v11;
  }
  else
  {
    v4 = 0;
    if ( v12 <= 1 )
      goto LABEL_8;
    v5 = v9;
    if ( v10 > 0x40 )
      v5 = (_QWORD *)*v9;
    v6 = v11;
    if ( v12 <= 0x40 )
    {
      v7 = (__int64)((_QWORD)v11 << (64 - (unsigned __int8)v12)) >> (64 - (unsigned __int8)v12);
      v4 = ~(_DWORD)v7;
      LOBYTE(v4) = v7 >= 0 && (unsigned __int64)v5 >= v7;
      if ( !(_BYTE)v4 )
      {
LABEL_8:
        if ( v10 <= 0x40 )
          return v4;
        goto LABEL_9;
      }
    }
    else
    {
      v7 = *v11;
      LOBYTE(v4) = *v11 >= 0 && *v11 <= (unsigned __int64)v5;
      if ( !(_BYTE)v4 )
        goto LABEL_15;
    }
    LOBYTE(v4) = (unsigned __int64)v5 - v7 >= a3 >> 3;
    if ( v12 <= 0x40 )
      goto LABEL_8;
  }
  if ( !v6 )
    goto LABEL_8;
LABEL_15:
  j_j___libc_free_0_0(v6);
  if ( v10 <= 0x40 )
    return v4;
LABEL_9:
  if ( v9 )
    j_j___libc_free_0_0(v9);
  return v4;
}
