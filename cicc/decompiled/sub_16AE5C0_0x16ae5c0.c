// Function: sub_16AE5C0
// Address: 0x16ae5c0
//
unsigned __int64 __fastcall sub_16AE5C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v7; // edi
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // r10
  unsigned int v10; // edx
  unsigned __int64 v11; // r9
  unsigned int v12; // ecx
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // ecx
  unsigned __int64 v16; // rdx
  unsigned int v17; // eax
  unsigned int v18; // ecx
  unsigned int v20; // eax
  unsigned int v21; // ecx
  unsigned int v22; // eax
  unsigned int v23; // ecx
  unsigned __int64 v24; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v25; // [rsp+8h] [rbp-68h]
  unsigned __int64 v26; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-58h]
  unsigned __int64 v28; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-48h]
  unsigned __int64 v30; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-38h]

  v7 = *(_DWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)a1;
  if ( v7 > 0x40 )
    v9 = *(_QWORD *)(v8 + 8LL * ((v7 - 1) >> 6));
  v10 = *(_DWORD *)(a2 + 8);
  v11 = *(_QWORD *)a2;
  v12 = v10 - 1;
  v13 = 1LL << ((unsigned __int8)v10 - 1);
  if ( (v9 & (1LL << ((unsigned __int8)v7 - 1))) != 0 )
  {
    if ( v10 > 0x40 )
    {
      if ( (*(_QWORD *)(v11 + 8LL * (v12 >> 6)) & v13) != 0 )
      {
        v29 = *(_DWORD *)(a2 + 8);
        sub_16A4FD0((__int64)&v28, (const void **)a2);
        LOBYTE(v10) = v29;
        if ( v29 > 0x40 )
        {
          sub_16A8F40((__int64 *)&v28);
LABEL_8:
          sub_16A7400((__int64)&v28);
          v14 = v29;
          v15 = *(_DWORD *)(a1 + 8);
          v29 = 0;
          v31 = v14;
          v25 = v15;
          v30 = v28;
          if ( v15 > 0x40 )
          {
            sub_16A4FD0((__int64)&v24, (const void **)a1);
            LOBYTE(v15) = v25;
            if ( v25 > 0x40 )
            {
              sub_16A8F40((__int64 *)&v24);
              goto LABEL_11;
            }
            v16 = v24;
          }
          else
          {
            v16 = *(_QWORD *)a1;
          }
          v24 = ~v16 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v15);
LABEL_11:
          sub_16A7400((__int64)&v24);
          v17 = v25;
          v25 = 0;
          v27 = v17;
          v26 = v24;
          sub_16ADD10((__int64)&v26, (__int64)&v30, (unsigned __int64 *)a3, (unsigned __int64 *)a4);
          if ( v27 > 0x40 && v26 )
            j_j___libc_free_0_0(v26);
          if ( v25 > 0x40 && v24 )
            j_j___libc_free_0_0(v24);
          if ( v31 > 0x40 && v30 )
            j_j___libc_free_0_0(v30);
          if ( v29 > 0x40 )
          {
            if ( v28 )
              j_j___libc_free_0_0(v28);
          }
          v18 = *(_DWORD *)(a4 + 8);
          if ( v18 <= 0x40 )
          {
LABEL_24:
            *(_QWORD *)a4 = ~*(_QWORD *)a4 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v18);
            return sub_16A7400(a4);
          }
LABEL_54:
          sub_16A8F40((__int64 *)a4);
          return sub_16A7400(a4);
        }
        v11 = v28;
LABEL_7:
        v28 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v10) & ~v11;
        goto LABEL_8;
      }
    }
    else if ( (v13 & v11) != 0 )
    {
      v29 = *(_DWORD *)(a2 + 8);
      goto LABEL_7;
    }
    v29 = v7;
    if ( v7 > 0x40 )
    {
      sub_16A4FD0((__int64)&v28, (const void **)a1);
      LOBYTE(v7) = v29;
      if ( v29 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v28);
        goto LABEL_45;
      }
      v8 = v28;
    }
    v28 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & ~v8;
LABEL_45:
    sub_16A7400((__int64)&v28);
    v22 = v29;
    v29 = 0;
    v31 = v22;
    v30 = v28;
    sub_16ADD10((__int64)&v30, a2, (unsigned __int64 *)a3, (unsigned __int64 *)a4);
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    if ( v29 > 0x40 && v28 )
      j_j___libc_free_0_0(v28);
    v23 = *(_DWORD *)(a3 + 8);
    if ( v23 > 0x40 )
      sub_16A8F40((__int64 *)a3);
    else
      *(_QWORD *)a3 = ~*(_QWORD *)a3 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v23);
    sub_16A7400(a3);
    v18 = *(_DWORD *)(a4 + 8);
    if ( v18 <= 0x40 )
      goto LABEL_24;
    goto LABEL_54;
  }
  if ( v10 <= 0x40 )
  {
    if ( (v13 & v11) != 0 )
    {
      v29 = *(_DWORD *)(a2 + 8);
LABEL_30:
      v28 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v10) & ~v11;
      goto LABEL_31;
    }
    return sub_16ADD10(a1, a2, (unsigned __int64 *)a3, (unsigned __int64 *)a4);
  }
  if ( (*(_QWORD *)(v11 + 8LL * (v12 >> 6)) & v13) == 0 )
    return sub_16ADD10(a1, a2, (unsigned __int64 *)a3, (unsigned __int64 *)a4);
  v29 = *(_DWORD *)(a2 + 8);
  sub_16A4FD0((__int64)&v28, (const void **)a2);
  LOBYTE(v10) = v29;
  if ( v29 <= 0x40 )
  {
    v11 = v28;
    goto LABEL_30;
  }
  sub_16A8F40((__int64 *)&v28);
LABEL_31:
  sub_16A7400((__int64)&v28);
  v20 = v29;
  v29 = 0;
  v31 = v20;
  v30 = v28;
  sub_16ADD10(a1, (__int64)&v30, (unsigned __int64 *)a3, (unsigned __int64 *)a4);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  v21 = *(_DWORD *)(a3 + 8);
  if ( v21 > 0x40 )
    sub_16A8F40((__int64 *)a3);
  else
    *(_QWORD *)a3 = ~*(_QWORD *)a3 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v21);
  return sub_16A7400(a3);
}
