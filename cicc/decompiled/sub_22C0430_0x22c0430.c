// Function: sub_22C0430
// Address: 0x22c0430
//
void __fastcall sub_22C0430(__int64 a1, unsigned __int8 *a2)
{
  int v2; // eax
  unsigned int v3; // eax
  unsigned __int64 v4; // rdx
  unsigned int v5; // eax
  unsigned __int64 v6; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-78h]
  unsigned __int64 v8; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-68h]
  unsigned __int64 v10; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v11; // [rsp+28h] [rbp-58h]
  unsigned __int64 v12; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+38h] [rbp-48h]
  unsigned __int64 v14; // [rsp+40h] [rbp-40h]
  unsigned int v15; // [rsp+48h] [rbp-38h]

  v2 = *a2;
  if ( (_BYTE)v2 == 17 )
  {
    v3 = *((_DWORD *)a2 + 8);
    v11 = v3;
    if ( v3 > 0x40 )
    {
      sub_C43780((__int64)&v10, (const void **)a2 + 3);
      v7 = *((_DWORD *)a2 + 8);
      if ( v7 > 0x40 )
      {
        sub_C43780((__int64)&v6, (const void **)a2 + 3);
LABEL_5:
        sub_C46A40((__int64)&v6, 1);
        v5 = v7;
        v7 = 0;
        v9 = v5;
        v8 = v6;
        sub_AADC30((__int64)&v12, (__int64)&v8, (__int64 *)&v10);
        sub_22C00F0(a1, (__int64)&v12, 0, 0, 1u);
        if ( v15 > 0x40 && v14 )
          j_j___libc_free_0_0(v14);
        if ( v13 > 0x40 && v12 )
          j_j___libc_free_0_0(v12);
        if ( v9 > 0x40 && v8 )
          j_j___libc_free_0_0(v8);
        if ( v7 > 0x40 && v6 )
          j_j___libc_free_0_0(v6);
        if ( v11 > 0x40 )
        {
          if ( v10 )
            j_j___libc_free_0_0(v10);
        }
        return;
      }
    }
    else
    {
      v4 = *((_QWORD *)a2 + 3);
      v7 = v3;
      v10 = v4;
    }
    v6 = *((_QWORD *)a2 + 3);
    goto LABEL_5;
  }
  if ( (unsigned int)(v2 - 12) > 1 && *(_BYTE *)a1 != 3 )
  {
    *(_BYTE *)a1 = 3;
    *(_QWORD *)(a1 + 8) = a2;
  }
}
