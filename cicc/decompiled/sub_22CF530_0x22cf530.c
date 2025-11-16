// Function: sub_22CF530
// Address: 0x22cf530
//
__int64 __fastcall sub_22CF530(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  char v12; // r13
  unsigned int v13; // eax
  unsigned int v14; // eax
  unsigned int v15; // esi
  unsigned int v17; // eax
  _BYTE v18[8]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v19; // [rsp+18h] [rbp-58h] BYREF
  unsigned int v20; // [rsp+20h] [rbp-50h]
  unsigned __int64 v21; // [rsp+28h] [rbp-48h] BYREF
  unsigned int v22; // [rsp+30h] [rbp-40h]

  v10 = sub_AA4B30(a4);
  v11 = sub_22C1480(a2, v10);
  sub_22CF010((__int64)v18, v11, a3, a4, a5, a6);
  v12 = v18[0];
  if ( (unsigned __int8)(v18[0] - 4) <= 1u )
  {
    v13 = v20;
    *(_DWORD *)(a1 + 8) = v20;
    if ( v13 > 0x40 )
    {
      sub_C43780(a1, (const void **)&v19);
      v17 = v22;
      *(_DWORD *)(a1 + 24) = v22;
      if ( v17 <= 0x40 )
      {
LABEL_4:
        *(_QWORD *)(a1 + 16) = v21;
        if ( (unsigned int)v18[0] - 4 > 1 )
          return a1;
        goto LABEL_5;
      }
    }
    else
    {
      *(_QWORD *)a1 = v19;
      v14 = v22;
      *(_DWORD *)(a1 + 24) = v22;
      if ( v14 <= 0x40 )
        goto LABEL_4;
    }
    sub_C43780(a1 + 16, (const void **)&v21);
    goto LABEL_11;
  }
  if ( v18[0] == 2 )
  {
    sub_AD8380(a1, v19);
LABEL_11:
    if ( (unsigned int)v18[0] - 4 > 1 )
      return a1;
    goto LABEL_14;
  }
  v15 = sub_BCB060(*(_QWORD *)(a3 + 8));
  if ( v12 )
  {
    sub_AADB10(a1, v15, 1);
    goto LABEL_11;
  }
  sub_AADB10(a1, v15, 0);
  if ( (unsigned int)v18[0] - 4 > 1 )
    return a1;
LABEL_14:
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
LABEL_5:
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return a1;
}
