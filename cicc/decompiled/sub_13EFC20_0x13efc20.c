// Function: sub_13EFC20
// Address: 0x13efc20
//
__int64 __fastcall sub_13EFC20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int *a5, __int64 a6)
{
  unsigned __int8 v7; // al
  unsigned int v9; // eax
  unsigned int v10; // eax
  __int64 v11; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-78h]
  __int64 v13; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-68h]
  __int64 v15; // [rsp+20h] [rbp-60h]
  unsigned int v16; // [rsp+28h] [rbp-58h]
  unsigned int v17; // [rsp+30h] [rbp-50h] BYREF
  __int64 v18; // [rsp+38h] [rbp-48h] BYREF
  unsigned int v19; // [rsp+40h] [rbp-40h]
  __int64 v20; // [rsp+48h] [rbp-38h] BYREF
  unsigned int v21; // [rsp+50h] [rbp-30h]

  v7 = *(_BYTE *)(a2 + 16);
  if ( v7 > 0x10u )
    return sub_13EEDA0(a1, (unsigned __int8 *)a2, a3, a4, a5, a6);
  v17 = 0;
  if ( v7 != 9 )
  {
    if ( v7 != 13 )
    {
      v17 = 1;
      v18 = a2;
      goto LABEL_3;
    }
    v12 = *(_DWORD *)(a2 + 32);
    if ( v12 > 0x40 )
      sub_16A4FD0(&v11, a2 + 24);
    else
      v11 = *(_QWORD *)(a2 + 24);
    sub_1589870(&v13, &v11);
    if ( v17 == 3 )
    {
      if ( !(unsigned __int8)sub_158A120(&v13) )
      {
        if ( v19 > 0x40 && v18 )
          j_j___libc_free_0_0(v18);
        v18 = v13;
        v10 = v14;
        v14 = 0;
        v19 = v10;
        if ( v21 <= 0x40 || !v20 )
        {
          v20 = v15;
          v21 = v16;
          goto LABEL_11;
        }
        j_j___libc_free_0_0(v20);
        v9 = v14;
        v20 = v15;
        v21 = v16;
LABEL_31:
        if ( v9 > 0x40 && v13 )
          j_j___libc_free_0_0(v13);
        goto LABEL_11;
      }
      if ( v17 == 4 )
      {
LABEL_27:
        if ( v16 > 0x40 && v15 )
          j_j___libc_free_0_0(v15);
        v9 = v14;
        goto LABEL_31;
      }
      if ( v17 - 1 > 1 )
      {
        if ( v17 == 3 )
        {
          sub_135E100(&v20);
          sub_135E100(&v18);
        }
        goto LABEL_26;
      }
    }
    else
    {
      if ( !(unsigned __int8)sub_158A120(&v13) )
      {
        v17 = 3;
        v19 = v14;
        v18 = v13;
        v21 = v16;
        v20 = v15;
LABEL_11:
        if ( v12 > 0x40 && v11 )
          j_j___libc_free_0_0(v11);
        goto LABEL_3;
      }
      if ( v17 == 4 )
        goto LABEL_27;
      if ( v17 - 1 > 1 )
      {
        if ( v17 == 3 )
        {
          if ( v21 > 0x40 && v20 )
            j_j___libc_free_0_0(v20);
          if ( v19 > 0x40 && v18 )
            j_j___libc_free_0_0(v18);
        }
        goto LABEL_26;
      }
    }
    v18 = 0;
LABEL_26:
    v17 = 4;
    goto LABEL_27;
  }
LABEL_3:
  sub_13E8810(a5, &v17);
  if ( v17 == 3 )
  {
    if ( v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    if ( v19 > 0x40 )
    {
      if ( v18 )
        j_j___libc_free_0_0(v18);
    }
  }
  return 1;
}
