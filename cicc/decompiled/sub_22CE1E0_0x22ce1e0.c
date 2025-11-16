// Function: sub_22CE1E0
// Address: 0x22ce1e0
//
__int64 __fastcall sub_22CE1E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v7; // r14
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  char v10; // bl
  unsigned int v11; // eax
  unsigned int v12; // esi
  unsigned int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // eax
  __int64 *v17; // rax
  unsigned int v19; // [rsp+Ch] [rbp-64h]
  _BYTE v20[8]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v21; // [rsp+18h] [rbp-58h] BYREF
  unsigned int v22; // [rsp+20h] [rbp-50h]
  unsigned __int64 v23; // [rsp+28h] [rbp-48h] BYREF
  unsigned int v24; // [rsp+30h] [rbp-40h]

  v7 = *(_QWORD *)(a4 + 40);
  v8 = sub_AA4B30(v7);
  v9 = sub_22C1480(a2, v8);
  sub_22CDEF0((__int64)v20, v9, a3, v7, a4);
  v10 = v20[0];
  if ( v20[0] == 4
    || (v11 = sub_BCB060(*(_QWORD *)(a3 + 8)), v12 = v11, v10 == 5)
    && (a5 || (v19 = v11, v17 = sub_9876C0(&v21), v10 = v20[0], v12 = v19, v17)) )
  {
    v14 = v22;
    *(_DWORD *)(a1 + 8) = v22;
    if ( v14 > 0x40 )
    {
      sub_C43780(a1, (const void **)&v21);
      v16 = v24;
      *(_DWORD *)(a1 + 24) = v24;
      if ( v16 <= 0x40 )
      {
LABEL_11:
        *(_QWORD *)(a1 + 16) = v23;
        if ( (unsigned int)v20[0] - 4 > 1 )
          return a1;
        goto LABEL_12;
      }
    }
    else
    {
      *(_QWORD *)a1 = v21;
      v15 = v24;
      *(_DWORD *)(a1 + 24) = v24;
      if ( v15 <= 0x40 )
        goto LABEL_11;
    }
    sub_C43780(a1 + 16, (const void **)&v23);
    goto LABEL_6;
  }
  if ( v10 == 2 )
  {
    sub_AD8380(a1, v21);
LABEL_6:
    if ( (unsigned int)v20[0] - 4 > 1 )
      return a1;
    goto LABEL_16;
  }
  if ( v10 )
  {
    sub_AADB10(a1, v12, 1);
    goto LABEL_6;
  }
  sub_AADB10(a1, v12, 0);
  if ( (unsigned int)v20[0] - 4 > 1 )
    return a1;
LABEL_16:
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
LABEL_12:
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return a1;
}
