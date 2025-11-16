// Function: sub_140C3C0
// Address: 0x140c3c0
//
__int64 __fastcall sub_140C3C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  _QWORD *v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned int v10; // eax
  __int64 v12; // rax
  unsigned int v13; // ecx
  unsigned __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // rsi
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned int v19; // r14d
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned int v22; // ebx
  unsigned __int64 v23; // r14
  char v24; // [rsp+Fh] [rbp-81h] BYREF
  unsigned __int64 v25; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-78h]
  unsigned __int64 v27; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-68h]
  unsigned __int64 v29; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-58h]
  __m128i v31; // [rsp+40h] [rbp-50h] BYREF
  char v32; // [rsp+50h] [rbp-40h]

  v4 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  sub_140AD80(&v31, a3 & 0xFFFFFFFFFFFFFFF8LL, *(_QWORD **)(a2 + 8));
  if ( !v32 )
    goto LABEL_20;
  v5 = (_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
  if ( v31.m128i_i8[0] != 16 )
  {
    v6 = v5[3 * v31.m128i_u32[2]];
    if ( *(_BYTE *)(v6 + 16) == 13 )
    {
      v26 = *(_DWORD *)(v6 + 32);
      if ( v26 > 0x40 )
        sub_16A4FD0(&v25, v6 + 24);
      else
        v25 = *(_QWORD *)(v6 + 24);
      if ( (unsigned __int8)sub_140B890(a2, &v25) )
      {
        if ( v31.m128i_i32[3] < 0 )
        {
          v20 = v26;
          *(_DWORD *)(a1 + 8) = v26;
          if ( v20 > 0x40 )
            sub_16A4FD0(a1, &v25);
          else
            *(_QWORD *)a1 = v25;
          v21 = *(_DWORD *)(a2 + 32);
          *(_DWORD *)(a1 + 24) = v21;
          if ( v21 > 0x40 )
            sub_16A4FD0(a1 + 16, a2 + 24);
          else
            *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 24);
          goto LABEL_25;
        }
        v7 = *(_QWORD *)(v4 + 24 * (v31.m128i_i32[3] - (unsigned __int64)(*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v7 + 16) == 13 )
        {
          v28 = *(_DWORD *)(v7 + 32);
          if ( v28 > 0x40 )
            sub_16A4FD0(&v27, v7 + 24);
          else
            v27 = *(_QWORD *)(v7 + 24);
          if ( !(unsigned __int8)sub_140B890(a2, &v27) )
            goto LABEL_23;
          sub_16AA580(&v29, &v25, &v27, &v24);
          if ( v26 > 0x40 && v25 )
            j_j___libc_free_0_0(v25);
          v8 = v29;
          v9 = v30;
          v25 = v29;
          v26 = v30;
          if ( v24 )
          {
LABEL_23:
            *(_DWORD *)(a1 + 8) = 1;
            *(_QWORD *)a1 = 0;
            *(_DWORD *)(a1 + 24) = 1;
            *(_QWORD *)(a1 + 16) = 0;
          }
          else
          {
            *(_DWORD *)(a1 + 8) = v30;
            if ( v9 > 0x40 )
              sub_16A4FD0(a1, &v25);
            else
              *(_QWORD *)a1 = v8;
            v10 = *(_DWORD *)(a2 + 32);
            *(_DWORD *)(a1 + 24) = v10;
            if ( v10 > 0x40 )
              sub_16A4FD0(a1 + 16, a2 + 24);
            else
              *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 24);
          }
          if ( v28 > 0x40 && v27 )
            j_j___libc_free_0_0(v27);
LABEL_25:
          if ( v26 <= 0x40 )
            return a1;
          goto LABEL_26;
        }
      }
      *(_DWORD *)(a1 + 8) = 1;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 24) = 1;
      *(_QWORD *)(a1 + 16) = 0;
      goto LABEL_25;
    }
LABEL_20:
    *(_DWORD *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = 0;
    return a1;
  }
  v12 = sub_14AD030(*v5, 8);
  v13 = *(_DWORD *)(a2 + 20);
  v26 = v13;
  if ( v13 <= 0x40 )
  {
    v14 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v13) & v12;
    v25 = v14;
    goto LABEL_31;
  }
  sub_16A4EF0(&v25, v12, 0);
  v19 = v26;
  if ( v26 <= 0x40 )
  {
    v14 = v25;
LABEL_31:
    v15 = v26;
    if ( v14 )
      goto LABEL_32;
LABEL_47:
    *(_DWORD *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_43;
  }
  v15 = sub_16A57B0(&v25);
  if ( v19 == v15 )
    goto LABEL_47;
LABEL_32:
  if ( v31.m128i_i32[2] > 0 )
  {
    v16 = *(_QWORD *)(v4 + 24 * (v31.m128i_i32[2] - (unsigned __int64)(*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v16 + 16) != 13 )
    {
      *(_DWORD *)(a1 + 8) = 1;
      v15 = v26;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 24) = 1;
      *(_QWORD *)(a1 + 16) = 0;
      goto LABEL_43;
    }
    sub_16A5DD0(&v27, v16 + 24, *(unsigned int *)(a2 + 20));
    if ( (int)sub_16A9900(&v25, &v27) > 0 )
    {
      v30 = v28;
      if ( v28 > 0x40 )
        sub_16A4FD0(&v29, &v27);
      else
        v29 = v27;
      sub_16A7490(&v29, 1);
      v22 = v30;
      v30 = 0;
      v23 = v29;
      if ( v26 > 0x40 && v25 )
      {
        j_j___libc_free_0_0(v25);
        v25 = v23;
        v26 = v22;
        if ( v30 > 0x40 && v29 )
          j_j___libc_free_0_0(v29);
      }
      else
      {
        v25 = v29;
        v26 = v22;
      }
    }
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
  }
  v17 = v26;
  *(_DWORD *)(a1 + 8) = v26;
  if ( v17 > 0x40 )
    sub_16A4FD0(a1, &v25);
  else
    *(_QWORD *)a1 = v25;
  v18 = *(_DWORD *)(a2 + 32);
  *(_DWORD *)(a1 + 24) = v18;
  if ( v18 > 0x40 )
    sub_16A4FD0(a1 + 16, a2 + 24);
  else
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 24);
  v15 = v26;
LABEL_43:
  if ( v15 <= 0x40 )
    return a1;
LABEL_26:
  if ( v25 )
    j_j___libc_free_0_0(v25);
  return a1;
}
