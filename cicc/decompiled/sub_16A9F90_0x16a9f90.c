// Function: sub_16A9F90
// Address: 0x16a9f90
//
__int64 __fastcall sub_16A9F90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r10
  unsigned int v8; // r8d
  unsigned __int64 v9; // r9
  unsigned int v10; // ecx
  __int64 v11; // rdx
  unsigned int v12; // eax
  unsigned int v13; // ecx
  unsigned __int64 v14; // rdx
  unsigned int v15; // eax
  unsigned __int64 v16; // rdi
  unsigned int v17; // eax
  char v18; // cl
  bool v19; // cc
  unsigned int v21; // eax
  unsigned __int64 v22; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v23; // [rsp+8h] [rbp-58h]
  unsigned __int64 v24; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-48h]
  unsigned __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-38h]
  unsigned __int64 v28; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-28h]

  v5 = *(_DWORD *)(a2 + 8);
  v6 = *(_QWORD *)a2;
  v7 = *(_QWORD *)a2;
  if ( v5 > 0x40 )
    v7 = *(_QWORD *)(v6 + 8LL * ((v5 - 1) >> 6));
  v8 = *(_DWORD *)(a3 + 8);
  v9 = *(_QWORD *)a3;
  v10 = v8 - 1;
  v11 = 1LL << ((unsigned __int8)v8 - 1);
  if ( (v7 & (1LL << ((unsigned __int8)v5 - 1))) != 0 )
  {
    if ( v8 > 0x40 )
    {
      if ( (v11 & *(_QWORD *)(v9 + 8LL * (v10 >> 6))) != 0 )
      {
LABEL_6:
        v23 = *(_DWORD *)(a2 + 8);
        if ( v5 > 0x40 )
        {
          sub_16A4FD0((__int64)&v22, (const void **)a2);
          LOBYTE(v5) = v23;
          if ( v23 > 0x40 )
          {
            sub_16A8F40((__int64 *)&v22);
LABEL_8:
            sub_16A7400((__int64)&v22);
            v12 = v23;
            v13 = *(_DWORD *)(a3 + 8);
            v23 = 0;
            v25 = v12;
            v27 = v13;
            v24 = v22;
            if ( v13 > 0x40 )
            {
              sub_16A4FD0((__int64)&v26, (const void **)a3);
              LOBYTE(v13) = v27;
              if ( v27 > 0x40 )
              {
                sub_16A8F40((__int64 *)&v26);
                goto LABEL_11;
              }
              v14 = v26;
            }
            else
            {
              v14 = *(_QWORD *)a3;
            }
            v26 = ~v14 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13);
LABEL_11:
            sub_16A7400((__int64)&v26);
            v15 = v27;
            v27 = 0;
            v29 = v15;
            v28 = v26;
            sub_16A9D70(a1, (__int64)&v24, (__int64)&v28);
            if ( v29 > 0x40 && v28 )
              j_j___libc_free_0_0(v28);
            if ( v27 > 0x40 && v26 )
              j_j___libc_free_0_0(v26);
            if ( v25 > 0x40 && v24 )
              j_j___libc_free_0_0(v24);
            if ( v23 <= 0x40 )
              return a1;
            v16 = v22;
            if ( !v22 )
              return a1;
LABEL_34:
            j_j___libc_free_0_0(v16);
            return a1;
          }
          v6 = v22;
        }
        v22 = ~v6 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v5);
        goto LABEL_8;
      }
    }
    else if ( (v11 & v9) != 0 )
    {
      goto LABEL_6;
    }
    v25 = *(_DWORD *)(a2 + 8);
    if ( v5 > 0x40 )
    {
      sub_16A4FD0((__int64)&v24, (const void **)a2);
      LOBYTE(v5) = v25;
      if ( v25 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v24);
LABEL_41:
        sub_16A7400((__int64)&v24);
        v21 = v25;
        v25 = 0;
        v27 = v21;
        v26 = v24;
        sub_16A9D70((__int64)&v28, (__int64)&v26, a3);
        v18 = v29;
        if ( v29 > 0x40 )
          goto LABEL_42;
        goto LABEL_28;
      }
      v6 = v24;
    }
    v24 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v6;
    goto LABEL_41;
  }
  if ( v8 <= 0x40 )
  {
    if ( (v11 & v9) != 0 )
    {
      v25 = v8;
LABEL_26:
      v24 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v8) & ~v9;
      goto LABEL_27;
    }
LABEL_36:
    sub_16A9D70(a1, a2, a3);
    return a1;
  }
  if ( (*(_QWORD *)(v9 + 8LL * (v10 >> 6)) & v11) == 0 )
    goto LABEL_36;
  v25 = *(_DWORD *)(a3 + 8);
  sub_16A4FD0((__int64)&v24, (const void **)a3);
  LOBYTE(v8) = v25;
  if ( v25 <= 0x40 )
  {
    v9 = v24;
    goto LABEL_26;
  }
  sub_16A8F40((__int64 *)&v24);
LABEL_27:
  sub_16A7400((__int64)&v24);
  v17 = v25;
  v25 = 0;
  v27 = v17;
  v26 = v24;
  sub_16A9D70((__int64)&v28, a2, (__int64)&v26);
  v18 = v29;
  if ( v29 > 0x40 )
  {
LABEL_42:
    sub_16A8F40((__int64 *)&v28);
    goto LABEL_29;
  }
LABEL_28:
  v28 = ~v28 & (0xFFFFFFFFFFFFFFFFLL >> -v18);
LABEL_29:
  sub_16A7400((__int64)&v28);
  v19 = v27 <= 0x40;
  *(_DWORD *)(a1 + 8) = v29;
  *(_QWORD *)a1 = v28;
  if ( !v19 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 )
  {
    v16 = v24;
    if ( v24 )
      goto LABEL_34;
  }
  return a1;
}
