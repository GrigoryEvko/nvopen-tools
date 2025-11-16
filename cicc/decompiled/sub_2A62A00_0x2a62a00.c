// Function: sub_2A62A00
// Address: 0x2a62a00
//
__int64 __fastcall sub_2A62A00(__int64 a1, unsigned __int8 *a2)
{
  int v2; // eax
  unsigned int v3; // eax
  unsigned __int64 v4; // rdx
  unsigned int v5; // eax
  unsigned int v6; // eax
  unsigned int v7; // r13d
  unsigned __int64 v9; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-78h]
  unsigned __int64 v11; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-68h]
  unsigned __int64 v13; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-58h]
  unsigned __int64 v15; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-48h]
  unsigned __int64 v17; // [rsp+40h] [rbp-40h]
  unsigned int v18; // [rsp+48h] [rbp-38h]

  v2 = *a2;
  if ( (_BYTE)v2 == 17 )
  {
    v3 = *((_DWORD *)a2 + 8);
    v14 = v3;
    if ( v3 > 0x40 )
    {
      sub_C43780((__int64)&v13, (const void **)a2 + 3);
      v10 = *((_DWORD *)a2 + 8);
      if ( v10 > 0x40 )
      {
        sub_C43780((__int64)&v9, (const void **)a2 + 3);
LABEL_5:
        sub_C46A40((__int64)&v9, 1);
        v5 = v10;
        v10 = 0;
        v12 = v5;
        v11 = v9;
        sub_AADC30((__int64)&v15, (__int64)&v11, (__int64 *)&v13);
        LOBYTE(v6) = sub_2A62120((char *)a1, (__int64)&v15, 0, 0, 1u);
        v7 = v6;
        if ( v18 > 0x40 && v17 )
          j_j___libc_free_0_0(v17);
        if ( v16 > 0x40 && v15 )
          j_j___libc_free_0_0(v15);
        if ( v12 > 0x40 && v11 )
          j_j___libc_free_0_0(v11);
        if ( v10 > 0x40 && v9 )
          j_j___libc_free_0_0(v9);
        if ( v14 > 0x40 && v13 )
          j_j___libc_free_0_0(v13);
        return v7;
      }
    }
    else
    {
      v4 = *((_QWORD *)a2 + 3);
      v10 = v3;
      v13 = v4;
    }
    v9 = *((_QWORD *)a2 + 3);
    goto LABEL_5;
  }
  v7 = 0;
  if ( (unsigned int)(v2 - 12) > 1 && *(_BYTE *)a1 != 3 )
  {
    *(_BYTE *)a1 = 3;
    v7 = 1;
    *(_QWORD *)(a1 + 8) = a2;
  }
  return v7;
}
