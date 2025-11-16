// Function: sub_13E9B70
// Address: 0x13e9b70
//
__int64 __fastcall sub_13E9B70(__int64 a1, __int64 a2, int *a3, __int64 a4, __int64 a5)
{
  int v6; // eax
  __int64 result; // rax
  int *v8; // r12
  char v9; // r8
  __int64 v10; // rax
  unsigned int v11; // ebx
  __int64 v12; // rax
  char v13; // bl
  unsigned int v14; // r13d
  __int64 v15; // r12
  bool v16; // cc
  unsigned __int8 v17; // bl
  __int64 v18; // rax
  unsigned int v19; // r13d
  __int64 v20; // r12
  char v21; // bl
  unsigned int v22; // [rsp+Ch] [rbp-64h]
  unsigned int v23; // [rsp+Ch] [rbp-64h]
  __int64 v24; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+20h] [rbp-50h]
  unsigned int v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-38h]
  __int64 v30; // [rsp+40h] [rbp-30h]
  unsigned int v31; // [rsp+48h] [rbp-28h]

  v6 = *a3;
  if ( *a3 == 1 )
  {
    v10 = sub_14D7760(a1, *((_QWORD *)a3 + 1), a2, a4, a5);
    if ( *(_BYTE *)(v10 + 16) == 13 )
    {
      v11 = *(_DWORD *)(v10 + 32);
      if ( v11 <= 0x40 )
        return *(_QWORD *)(v10 + 24) != 0;
      return v11 != (unsigned int)sub_16A57B0(v10 + 24);
    }
    return 0xFFFFFFFFLL;
  }
  if ( v6 != 3 )
  {
    if ( v6 == 2 )
    {
      if ( (_DWORD)a1 == 32 )
      {
        v18 = sub_14D7760(33, *((_QWORD *)a3 + 1), a2, a4, a5);
        return -((unsigned __int8)sub_1593BB0(v18) ^ 1);
      }
      if ( (_DWORD)a1 == 33 )
      {
        v12 = sub_14D7760(a1, *((_QWORD *)a3 + 1), a2, a4, a5);
        if ( (unsigned __int8)sub_1593BB0(v12) )
          return 1;
      }
    }
    return 0xFFFFFFFFLL;
  }
  if ( *(_BYTE *)(a2 + 16) != 13 )
    return 0xFFFFFFFFLL;
  v8 = a3 + 2;
  if ( (_DWORD)a1 == 32 )
  {
    if ( (unsigned __int8)sub_158B950(v8, a2 + 24) )
    {
      v25 = a3[4];
      if ( v25 > 0x40 )
        sub_16A4FD0(&v24, v8);
      else
        v24 = *((_QWORD *)a3 + 1);
      sub_16A7490(&v24, 1);
      v19 = v25;
      v20 = v24;
      v25 = 0;
      v16 = (unsigned int)a3[8] <= 0x40;
      v29 = v19;
      v28 = v24;
      if ( v16 )
        v21 = *((_QWORD *)a3 + 3) == v24;
      else
        v21 = sub_16A5220(a3 + 6, &v28);
      if ( v19 > 0x40 )
      {
        if ( v20 )
        {
          j_j___libc_free_0_0(v20);
          if ( v25 > 0x40 )
          {
            if ( v24 )
              j_j___libc_free_0_0(v24);
          }
        }
      }
      if ( v21 )
        return 1;
      return 0xFFFFFFFFLL;
    }
    return 0;
  }
  if ( (_DWORD)a1 == 33 )
  {
    if ( !(unsigned __int8)sub_158B950(v8, a2 + 24) )
      return 1;
    v25 = a3[4];
    if ( v25 > 0x40 )
      sub_16A4FD0(&v24, v8);
    else
      v24 = *((_QWORD *)a3 + 1);
    sub_16A7490(&v24, 1);
    v14 = v25;
    v15 = v24;
    v25 = 0;
    v16 = (unsigned int)a3[8] <= 0x40;
    v29 = v14;
    v28 = v24;
    if ( v16 )
      v17 = *((_QWORD *)a3 + 3) == v24;
    else
      v17 = sub_16A5220(a3 + 6, &v28);
    if ( v14 > 0x40 )
    {
      if ( v15 )
      {
        j_j___libc_free_0_0(v15);
        if ( v25 > 0x40 )
        {
          if ( v24 )
            j_j___libc_free_0_0(v24);
        }
      }
    }
    return -(v17 ^ 1);
  }
  else
  {
    sub_158B890(&v24, (unsigned int)a1, a2 + 24);
    v9 = sub_158BB40(&v24, v8);
    result = 1;
    if ( !v9 )
    {
      sub_1590E70(&v28, &v24);
      v13 = sub_158BB40(&v28, v8);
      if ( v31 > 0x40 && v30 )
        j_j___libc_free_0_0(v30);
      if ( v29 > 0x40 && v28 )
        j_j___libc_free_0_0(v28);
      result = 0;
      if ( !v13 )
      {
        if ( v27 > 0x40 && v26 )
          j_j___libc_free_0_0(v26);
        if ( v25 > 0x40 && v24 )
          j_j___libc_free_0_0(v24);
        return 0xFFFFFFFFLL;
      }
    }
    if ( v27 > 0x40 && v26 )
    {
      v22 = result;
      j_j___libc_free_0_0(v26);
      result = v22;
    }
    if ( v25 > 0x40 && v24 )
    {
      v23 = result;
      j_j___libc_free_0_0(v24);
      return v23;
    }
  }
  return result;
}
