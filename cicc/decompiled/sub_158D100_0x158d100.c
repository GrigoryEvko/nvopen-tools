// Function: sub_158D100
// Address: 0x158d100
//
__int64 __fastcall sub_158D100(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v6; // eax
  __int64 v7; // rsi
  unsigned int v8; // ebx
  int v9; // ebx
  __int64 v10; // rdi
  unsigned int v12; // r15d
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rsi
  unsigned int v18; // eax
  unsigned int v19; // eax
  __int64 v20; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-58h]
  _QWORD *v22; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-48h]
  _QWORD *v24; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+38h] [rbp-38h]

  if ( sub_158A120(a2) )
  {
    sub_15897D0(a1, a3, 0);
    return a1;
  }
  v6 = *(_DWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 16);
  v8 = v6 - 1;
  if ( v6 <= 0x40 )
  {
    if ( v7 != 1LL << v8 )
      goto LABEL_5;
LABEL_9:
    sub_16A5C50(&v24, a2 + 16, a3);
    goto LABEL_10;
  }
  if ( (*(_QWORD *)(v7 + 8LL * (v8 >> 6)) & (1LL << v8)) != 0 && (unsigned int)sub_16A58A0(a2 + 16) == v8 )
    goto LABEL_9;
LABEL_5:
  v9 = *(_DWORD *)(a2 + 8);
  if ( sub_158A0B0(a2) || sub_158B9F0(a2) )
  {
    v23 = a3;
    v12 = v9 - 1;
    if ( a3 <= 0x40 )
    {
      v22 = 0;
      if ( v9 == 1 )
      {
        sub_16A7490(&v22, 1);
        v19 = v23;
        v21 = a3;
        v23 = 0;
        v25 = v19;
        v24 = v22;
        goto LABEL_43;
      }
      if ( v12 <= 0x40 )
      {
        v13 = 0;
        v14 = 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v9);
LABEL_41:
        v22 = (_QWORD *)(v13 | v14);
        goto LABEL_42;
      }
    }
    else
    {
      sub_16A4EF0(&v22, 0, 0);
      if ( v9 == 1 )
      {
LABEL_22:
        sub_16A7490(&v22, 1);
        v15 = v23;
        v21 = a3;
        v23 = 0;
        v25 = v15;
        v24 = v22;
LABEL_23:
        sub_16A4EF0(&v20, 0, 0);
        v16 = v21;
        goto LABEL_24;
      }
      if ( v12 <= 0x40 )
      {
        v13 = (unsigned __int64)v22;
        v14 = 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v9);
        if ( v23 > 0x40 )
        {
          *v22 |= v14;
          goto LABEL_22;
        }
        goto LABEL_41;
      }
    }
    sub_16A5260(&v22, 0, v12);
LABEL_42:
    sub_16A7490(&v22, 1);
    v18 = v23;
    v21 = a3;
    v23 = 0;
    v25 = v18;
    v24 = v22;
    if ( a3 > 0x40 )
      goto LABEL_23;
LABEL_43:
    v20 = 0;
    v16 = a3;
LABEL_24:
    v17 = ~a3 + v9 + (unsigned int)v16;
    if ( (_DWORD)v17 != (_DWORD)v16 )
    {
      if ( (unsigned int)v17 > 0x3F || (unsigned int)v16 > 0x40 )
        sub_16A5260(&v20, v17, v16);
      else
        v20 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v9 - (unsigned __int8)a3 + 63) << (~(_BYTE)a3
                                                                                          + v9
                                                                                          + (unsigned __int8)v16);
    }
    sub_15898E0(a1, (__int64)&v20, (__int64 *)&v24);
    if ( v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    if ( v25 > 0x40 && v24 )
      j_j___libc_free_0_0(v24);
    if ( v23 > 0x40 )
    {
      v10 = (__int64)v22;
      if ( v22 )
        goto LABEL_15;
    }
    return a1;
  }
  sub_16A5B10(&v24, a2 + 16, a3);
LABEL_10:
  sub_16A5B10(&v22, a2, a3);
  sub_15898E0(a1, (__int64)&v22, (__int64 *)&v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v25 > 0x40 )
  {
    v10 = (__int64)v24;
    if ( v24 )
LABEL_15:
      j_j___libc_free_0_0(v10);
  }
  return a1;
}
