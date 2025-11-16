// Function: sub_158DBC0
// Address: 0x158dbc0
//
__int64 __fastcall sub_158DBC0(__int64 a1, __int64 a2, int a3, unsigned int a4)
{
  unsigned int v7; // r8d
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // r14d
  __int64 v12; // rbx
  unsigned int v13; // eax
  unsigned __int64 v14; // rdi
  unsigned int v15; // ebx
  unsigned int v16; // eax
  unsigned int v17; // eax
  __int64 v18; // rbx
  unsigned int v19; // [rsp+4h] [rbp-7Ch]
  unsigned int v20; // [rsp+4h] [rbp-7Ch]
  char v21; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-68h]
  unsigned __int64 v24; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v27; // [rsp+38h] [rbp-48h]
  unsigned __int64 v28; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+48h] [rbp-38h]

  switch ( a3 )
  {
    case '$':
      sub_158D430(a1, a2, a4);
      return a1;
    case '%':
      sub_158CEA0(a1, a2, a4);
      return a1;
    case '&':
      sub_158D100(a1, a2, a4);
      return a1;
    case '\'':
    case '(':
      v7 = *(_DWORD *)(a2 + 8);
      if ( v7 == a4 )
      {
        *(_DWORD *)(a1 + 8) = a4;
        if ( a4 <= 0x40 )
          goto LABEL_9;
        goto LABEL_48;
      }
      sub_15897D0(a1, v7, 1);
      return a1;
    case ')':
      v15 = *(_DWORD *)(a2 + 8);
      v29 = v15;
      if ( v15 > 0x40 )
      {
        sub_16A4EF0(&v28, 0, 0);
        sub_16A5DD0(&v22, &v28, a4);
        if ( v29 > 0x40 && v28 )
          j_j___libc_free_0_0(v28);
        v29 = v15;
        sub_16A4EF0(&v28, -1, 1);
      }
      else
      {
        v28 = 0;
        sub_16A5DD0(&v22, &v28, a4);
        if ( v29 > 0x40 && v28 )
          j_j___libc_free_0_0(v28);
        v29 = v15;
        v28 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
      }
      sub_16A5DD0(&v24, &v28, a4);
      if ( v29 <= 0x40 )
        goto LABEL_23;
      goto LABEL_21;
    case '*':
      v10 = *(_DWORD *)(a2 + 8);
      v11 = v10 - 1;
      v29 = v10;
      v19 = v10;
      v21 = (v10 - 1) & 0x3F;
      v12 = 1LL << ((unsigned __int8)v10 - 1);
      if ( v10 > 0x40 )
      {
        sub_16A4EF0(&v28, 0, 0);
        if ( v29 <= 0x40 )
        {
          v28 |= v12;
          sub_16A5E20(&v22, &v28, a4);
          v13 = v19;
          if ( v29 > 0x40 )
            goto LABEL_14;
        }
        else
        {
          *(_QWORD *)(v28 + 8LL * (v11 >> 6)) |= v12;
          sub_16A5E20(&v22, &v28, a4);
          v13 = v19;
          if ( v29 > 0x40 )
          {
            v14 = v28;
            if ( v28 )
              goto LABEL_42;
          }
        }
        v29 = v13;
        v18 = ~v12;
        goto LABEL_51;
      }
      v28 = 1LL << ((unsigned __int8)v10 - 1);
      sub_16A5E20(&v22, &v28, a4);
      v13 = v19;
      if ( v29 <= 0x40 )
      {
        v29 = v19;
        v18 = ~v12;
        goto LABEL_44;
      }
LABEL_14:
      v14 = v28;
      if ( v28 )
      {
LABEL_42:
        v20 = v13;
        j_j___libc_free_0_0(v14);
        v13 = v20;
      }
      v29 = v13;
      v18 = ~v12;
      if ( v13 <= 0x40 )
      {
LABEL_44:
        v28 = 0xFFFFFFFFFFFFFFFFLL >> (63 - v21);
LABEL_45:
        v28 &= v18;
        goto LABEL_53;
      }
LABEL_51:
      sub_16A4EF0(&v28, -1, 1);
      if ( v29 <= 0x40 )
        goto LABEL_45;
      *(_QWORD *)(v28 + 8LL * (v11 >> 6)) &= v18;
LABEL_53:
      sub_16A5E20(&v24, &v28, a4);
      if ( v29 > 0x40 )
      {
LABEL_21:
        if ( v28 )
          j_j___libc_free_0_0(v28);
      }
LABEL_23:
      v16 = v25;
      v25 = 0;
      v29 = v16;
      v28 = v24;
      v17 = v23;
      v23 = 0;
      v27 = v17;
      v26 = v22;
      sub_15898E0(a1, (__int64)&v26, (__int64 *)&v28);
      if ( v27 > 0x40 && v26 )
        j_j___libc_free_0_0(v26);
      if ( v29 > 0x40 && v28 )
        j_j___libc_free_0_0(v28);
      if ( v25 > 0x40 && v24 )
        j_j___libc_free_0_0(v24);
      if ( v23 > 0x40 && v22 )
        j_j___libc_free_0_0(v22);
      return a1;
    case '+':
    case ',':
    case '-':
    case '.':
    case '0':
      sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
      return a1;
    case '/':
      v8 = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 8) = v8;
      if ( v8 > 0x40 )
LABEL_48:
        sub_16A4FD0(a1, a2);
      else
LABEL_9:
        *(_QWORD *)a1 = *(_QWORD *)a2;
      v9 = *(_DWORD *)(a2 + 24);
      *(_DWORD *)(a1 + 24) = v9;
      if ( v9 > 0x40 )
        sub_16A4FD0(a1 + 16, a2 + 16);
      else
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
      return a1;
  }
}
