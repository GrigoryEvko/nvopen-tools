// Function: sub_16AB4D0
// Address: 0x16ab4d0
//
__int64 __fastcall sub_16AB4D0(__int64 a1, __int64 a2, __int64 a3)
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
  bool v16; // cc
  unsigned __int64 v17; // rdi
  unsigned int v18; // eax
  unsigned int v20; // eax
  unsigned __int64 v21; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-68h]
  unsigned __int64 v23; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-58h]
  unsigned __int64 v25; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-48h]
  unsigned __int64 v27; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-38h]
  unsigned __int64 v29; // [rsp+40h] [rbp-30h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-28h]

  v5 = *(_DWORD *)(a2 + 8);
  v6 = *(_QWORD *)a2;
  v7 = *(_QWORD *)a2;
  if ( v5 > 0x40 )
    v7 = *(_QWORD *)(v6 + 8LL * ((v5 - 1) >> 6));
  v8 = *(_DWORD *)(a3 + 8);
  v9 = *(_QWORD *)a3;
  v10 = v8 - 1;
  v11 = 1LL << ((unsigned __int8)v8 - 1);
  if ( (v7 & (1LL << ((unsigned __int8)v5 - 1))) == 0 )
  {
    if ( v8 > 0x40 )
    {
      if ( (*(_QWORD *)(v9 + 8LL * (v10 >> 6)) & v11) != 0 )
      {
        v28 = *(_DWORD *)(a3 + 8);
        sub_16A4FD0((__int64)&v27, (const void **)a3);
        LOBYTE(v8) = v28;
        if ( v28 > 0x40 )
        {
          sub_16A8F40((__int64 *)&v27);
          goto LABEL_29;
        }
        v9 = v27;
LABEL_28:
        v27 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v8) & ~v9;
LABEL_29:
        sub_16A7400((__int64)&v27);
        v18 = v28;
        v28 = 0;
        v30 = v18;
        v29 = v27;
        sub_16AB0A0(a1, a2, (__int64)&v29);
        if ( v30 > 0x40 && v29 )
          j_j___libc_free_0_0(v29);
        if ( v28 > 0x40 )
        {
          v17 = v27;
          if ( v27 )
            goto LABEL_34;
        }
        return a1;
      }
    }
    else if ( (v11 & v9) != 0 )
    {
      v28 = v8;
      goto LABEL_28;
    }
    sub_16AB0A0(a1, a2, a3);
    return a1;
  }
  if ( v8 > 0x40 )
  {
    if ( (v11 & *(_QWORD *)(v9 + 8LL * (v10 >> 6))) != 0 )
    {
LABEL_6:
      v22 = *(_DWORD *)(a2 + 8);
      if ( v5 > 0x40 )
      {
        sub_16A4FD0((__int64)&v21, (const void **)a2);
        LOBYTE(v5) = v22;
        if ( v22 > 0x40 )
        {
          sub_16A8F40((__int64 *)&v21);
LABEL_8:
          sub_16A7400((__int64)&v21);
          v12 = v22;
          v13 = *(_DWORD *)(a3 + 8);
          v22 = 0;
          v24 = v12;
          v26 = v13;
          v23 = v21;
          if ( v13 > 0x40 )
          {
            sub_16A4FD0((__int64)&v25, (const void **)a3);
            LOBYTE(v13) = v26;
            if ( v26 > 0x40 )
            {
              sub_16A8F40((__int64 *)&v25);
LABEL_11:
              sub_16A7400((__int64)&v25);
              v15 = v26;
              v26 = 0;
              v28 = v15;
              v27 = v25;
              sub_16AB0A0((__int64)&v29, (__int64)&v23, (__int64)&v27);
              if ( v30 > 0x40 )
                sub_16A8F40((__int64 *)&v29);
              else
                v29 = ~v29 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v30);
              sub_16A7400((__int64)&v29);
              v16 = v28 <= 0x40;
              *(_DWORD *)(a1 + 8) = v30;
              *(_QWORD *)a1 = v29;
              if ( !v16 && v27 )
                j_j___libc_free_0_0(v27);
              if ( v26 > 0x40 && v25 )
                j_j___libc_free_0_0(v25);
              if ( v24 > 0x40 && v23 )
                j_j___libc_free_0_0(v23);
              if ( v22 <= 0x40 )
                return a1;
              v17 = v21;
              if ( !v21 )
                return a1;
LABEL_34:
              j_j___libc_free_0_0(v17);
              return a1;
            }
            v14 = v25;
          }
          else
          {
            v14 = *(_QWORD *)a3;
          }
          v25 = ~v14 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13);
          goto LABEL_11;
        }
        v6 = v21;
      }
      v21 = ~v6 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v5);
      goto LABEL_8;
    }
  }
  else if ( (v11 & v9) != 0 )
  {
    goto LABEL_6;
  }
  v26 = *(_DWORD *)(a2 + 8);
  if ( v5 <= 0x40 )
    goto LABEL_40;
  sub_16A4FD0((__int64)&v25, (const void **)a2);
  LOBYTE(v5) = v26;
  if ( v26 <= 0x40 )
  {
    v6 = v25;
LABEL_40:
    v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v6;
    goto LABEL_41;
  }
  sub_16A8F40((__int64 *)&v25);
LABEL_41:
  sub_16A7400((__int64)&v25);
  v20 = v26;
  v26 = 0;
  v28 = v20;
  v27 = v25;
  sub_16AB0A0((__int64)&v29, (__int64)&v27, a3);
  if ( v30 > 0x40 )
    sub_16A8F40((__int64 *)&v29);
  else
    v29 = ~v29 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v30);
  sub_16A7400((__int64)&v29);
  v16 = v28 <= 0x40;
  *(_DWORD *)(a1 + 8) = v30;
  *(_QWORD *)a1 = v29;
  if ( !v16 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 )
  {
    v17 = v25;
    if ( v25 )
      goto LABEL_34;
  }
  return a1;
}
