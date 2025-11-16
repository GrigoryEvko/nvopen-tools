// Function: sub_11559D0
// Address: 0x11559d0
//
__int64 __fastcall sub_11559D0(_BYTE **a1, __int64 a2, __int64 a3)
{
  char v4; // r13
  __int64 v6; // rdi
  _BYTE *v7; // rax
  unsigned __int64 *v8; // r13
  __int64 v9; // rdi
  _BYTE *v10; // rax
  _BYTE *v11; // rax
  unsigned int v12; // r14d
  int v13; // eax
  __int64 v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+8h] [rbp-58h]
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+8h] [rbp-58h]
  _BYTE v19[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v20; // [rsp+30h] [rbp-30h]

  v4 = *(_BYTE *)(*(_QWORD *)*a1 + 1LL) >> 1;
  if ( !*a1[1] )
  {
LABEL_11:
    if ( *a1[3] )
    {
      if ( (v4 & 1) != 0 )
      {
        v20 = 257;
        return sub_B504D0(19, a2, a3, (__int64)v19, 0, 0);
      }
      v8 = (unsigned __int64 *)(a2 + 24);
      if ( *(_BYTE *)a2 != 17 )
      {
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 > 1 )
          return 0;
        v16 = a3;
        if ( *(_BYTE *)a2 > 0x15u )
          return 0;
        v10 = sub_AD7630(a2, 0, a3);
        if ( !v10 || *v10 != 17 )
          return 0;
        a3 = v16;
        v8 = (unsigned __int64 *)(v10 + 24);
      }
      v9 = a3 + 24;
      if ( *(_BYTE *)a3 != 17 )
      {
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17 > 1 )
          return 0;
        if ( *(_BYTE *)a3 > 0x15u )
          return 0;
        v17 = a3;
        v11 = sub_AD7630(a3, 0, a3);
        if ( !v11 || *v11 != 17 )
          return 0;
        a3 = v17;
        v9 = (__int64)(v11 + 24);
      }
      v15 = a3;
      if ( (int)sub_C49970(v9, v8) <= 0 )
      {
        v20 = 257;
        return sub_B504D0(19, a2, v15, (__int64)v19, 0, 0);
      }
    }
    return 0;
  }
  if ( !*a1[2] || (v4 & 2) == 0 )
    return 0;
  v6 = a3 + 24;
  if ( *(_BYTE *)a3 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17 > 1 || *(_BYTE *)a3 > 0x15u )
      return 0;
    v14 = a3;
    v7 = sub_AD7630(a3, 0, a3);
    a3 = v14;
    if ( !v7 )
      goto LABEL_10;
    v6 = (__int64)(v7 + 24);
    if ( *v7 != 17 )
      goto LABEL_10;
  }
  v12 = *(_DWORD *)(v6 + 8);
  if ( !v12 )
  {
LABEL_10:
    if ( *a1[1] )
      return 0;
    goto LABEL_11;
  }
  if ( v12 <= 0x40 )
  {
    if ( *(_QWORD *)v6 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) )
      goto LABEL_31;
    goto LABEL_10;
  }
  v18 = a3;
  v13 = sub_C445E0(v6);
  a3 = v18;
  if ( v12 == v13 )
    goto LABEL_10;
LABEL_31:
  v20 = 257;
  return sub_B504D0(20, a2, a3, (__int64)v19, 0, 0);
}
