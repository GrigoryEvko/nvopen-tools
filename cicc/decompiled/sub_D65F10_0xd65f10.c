// Function: sub_D65F10
// Address: 0xd65f10
//
__int64 __fastcall sub_D65F10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 v6; // rdx
  __int64 *v7; // rbx
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // [rsp+10h] [rbp-F0h]
  __int64 v11; // [rsp+10h] [rbp-F0h]
  __int64 *v12; // [rsp+28h] [rbp-D8h]
  const void *v13; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v14; // [rsp+38h] [rbp-C8h]
  const void *v15; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v16; // [rsp+48h] [rbp-B8h]
  const void *v17; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v18; // [rsp+58h] [rbp-A8h]
  const void *v19; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v20; // [rsp+68h] [rbp-98h]
  const void *v21; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v22; // [rsp+78h] [rbp-88h]
  const void *v23; // [rsp+80h] [rbp-80h]
  unsigned int v24; // [rsp+88h] [rbp-78h]
  __int64 v25; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v26; // [rsp+98h] [rbp-68h]
  __int64 v27; // [rsp+A0h] [rbp-60h]
  unsigned int v28; // [rsp+A8h] [rbp-58h]
  const void *v29; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+B8h] [rbp-48h]
  const void *v31; // [rsp+C0h] [rbp-40h] BYREF
  unsigned int v32; // [rsp+C8h] [rbp-38h]

  if ( (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) != 0 )
  {
    v4 = 4LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    {
      v5 = *(__int64 **)(a3 - 8);
      v12 = &v5[v4];
    }
    else
    {
      v12 = (__int64 *)a3;
      v5 = (__int64 *)(a3 - v4 * 8);
    }
    v6 = *v5;
    v7 = v5 + 4;
    sub_D62600((__int64)&v13, a2, v6);
    for ( ; v7 != v12; v7 += 4 )
    {
      v8 = *v7;
      v18 = v14;
      if ( v14 > 0x40 )
      {
        v11 = v8;
        sub_C43780((__int64)&v17, &v13);
        v8 = v11;
      }
      else
      {
        v17 = v13;
      }
      v20 = v16;
      if ( v16 > 0x40 )
      {
        v10 = v8;
        sub_C43780((__int64)&v19, &v15);
        v8 = v10;
      }
      else
      {
        v19 = v15;
      }
      sub_D62600((__int64)&v25, a2, v8);
      v30 = v18;
      if ( v18 > 0x40 )
        sub_C43780((__int64)&v29, &v17);
      else
        v29 = v17;
      v32 = v20;
      if ( v20 > 0x40 )
        sub_C43780((__int64)&v31, &v19);
      else
        v31 = v19;
      sub_D5E640((__int64)&v21, a2, (__int64)&v29, (__int64)&v25);
      if ( v32 > 0x40 && v31 )
        j_j___libc_free_0_0(v31);
      if ( v30 > 0x40 && v29 )
        j_j___libc_free_0_0(v29);
      if ( v28 > 0x40 && v27 )
        j_j___libc_free_0_0(v27);
      if ( v26 > 0x40 && v25 )
        j_j___libc_free_0_0(v25);
      if ( v14 > 0x40 && v13 )
        j_j___libc_free_0_0(v13);
      v13 = v21;
      v9 = v22;
      v22 = 0;
      v14 = v9;
      if ( v16 > 0x40 && v15 )
      {
        j_j___libc_free_0_0(v15);
        v15 = v23;
        v16 = v24;
        if ( v22 > 0x40 && v21 )
          j_j___libc_free_0_0(v21);
      }
      else
      {
        v15 = v23;
        v16 = v24;
      }
      if ( v20 > 0x40 && v19 )
        j_j___libc_free_0_0(v19);
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
    }
    *(_DWORD *)(a1 + 8) = v14;
    *(_QWORD *)a1 = v13;
    *(_DWORD *)(a1 + 24) = v16;
    *(_QWORD *)(a1 + 16) = v15;
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 24) = 1;
  }
  return a1;
}
