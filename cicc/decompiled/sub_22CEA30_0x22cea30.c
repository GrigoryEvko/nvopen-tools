// Function: sub_22CEA30
// Address: 0x22cea30
//
__int64 __fastcall sub_22CEA30(__int64 a1, __int64 *a2, __int64 *a3, char a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int8 v8; // r14
  unsigned int v9; // eax
  unsigned int v10; // esi
  unsigned int v12; // eax
  unsigned int v13; // eax
  unsigned int v14; // eax
  __int64 *v15; // rax
  unsigned int v16; // [rsp+Ch] [rbp-54h]
  unsigned __int8 v17[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v18; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v19; // [rsp+20h] [rbp-40h]
  unsigned __int64 v20; // [rsp+28h] [rbp-38h] BYREF
  unsigned int v21; // [rsp+30h] [rbp-30h]

  v6 = sub_B43CA0(a3[3]);
  v7 = sub_22C1480(a2, v6);
  sub_22CE7B0(v17, v7, a3);
  v8 = v17[0];
  if ( v17[0] == 4
    || (v9 = sub_BCB060(*(_QWORD *)(*a3 + 8)), v10 = v9, v8 == 5)
    && (a4 || (v16 = v9, v15 = sub_9876C0(&v18), v8 = v17[0], v10 = v16, v15)) )
  {
    v12 = v19;
    *(_DWORD *)(a1 + 8) = v19;
    if ( v12 > 0x40 )
    {
      sub_C43780(a1, (const void **)&v18);
      v14 = v21;
      *(_DWORD *)(a1 + 24) = v21;
      if ( v14 <= 0x40 )
      {
LABEL_11:
        *(_QWORD *)(a1 + 16) = v20;
        if ( (unsigned int)v17[0] - 4 > 1 )
          return a1;
        goto LABEL_12;
      }
    }
    else
    {
      *(_QWORD *)a1 = v18;
      v13 = v21;
      *(_DWORD *)(a1 + 24) = v21;
      if ( v13 <= 0x40 )
        goto LABEL_11;
    }
    sub_C43780(a1 + 16, (const void **)&v20);
    goto LABEL_6;
  }
  if ( v8 == 2 )
  {
    sub_AD8380(a1, v18);
LABEL_6:
    if ( (unsigned int)v17[0] - 4 > 1 )
      return a1;
    goto LABEL_16;
  }
  if ( v8 )
  {
    sub_AADB10(a1, v10, 1);
    goto LABEL_6;
  }
  sub_AADB10(a1, v10, 0);
  if ( (unsigned int)v17[0] - 4 > 1 )
    return a1;
LABEL_16:
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
LABEL_12:
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return a1;
}
