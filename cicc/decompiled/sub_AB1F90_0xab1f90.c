// Function: sub_AB1F90
// Address: 0xab1f90
//
__int64 __fastcall sub_AB1F90(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r15
  unsigned int v5; // ebx
  unsigned int v6; // eax
  unsigned int v7; // eax
  __int64 v9; // rax
  unsigned int v10; // eax
  unsigned int v11; // eax
  __int64 v12; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-68h]
  __int64 v14; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-58h]
  __int64 v16; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-48h]
  __int64 v18; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-38h]

  v3 = a2 + 2;
  v5 = *((_DWORD *)a2 + 2);
  if ( v5 <= 0x40 )
  {
    v9 = *a2;
    if ( *a2 != a2[2] )
      goto LABEL_3;
    *(_DWORD *)(a1 + 8) = v5;
    *(_QWORD *)a1 = v9;
    v10 = *((_DWORD *)a2 + 6);
    *(_DWORD *)(a1 + 24) = v10;
    if ( v10 > 0x40 )
    {
LABEL_24:
      sub_C43780(a1 + 16, v3);
      return a1;
    }
LABEL_22:
    *(_QWORD *)(a1 + 16) = a2[2];
    return a1;
  }
  if ( (unsigned __int8)sub_C43C50(a2, a2 + 2) )
  {
    *(_DWORD *)(a1 + 8) = v5;
    sub_C43780(a1, a2);
    v11 = *((_DWORD *)a2 + 6);
    *(_DWORD *)(a1 + 24) = v11;
    if ( v11 > 0x40 )
      goto LABEL_24;
    goto LABEL_22;
  }
LABEL_3:
  v17 = *((_DWORD *)a2 + 6);
  if ( v17 > 0x40 )
    sub_C43780(&v16, v3);
  else
    v16 = a2[2];
  sub_C46B40(&v16, a3);
  v6 = v17;
  v17 = 0;
  v19 = v6;
  v18 = v16;
  v13 = *((_DWORD *)a2 + 2);
  if ( v13 > 0x40 )
    sub_C43780(&v12, a2);
  else
    v12 = *a2;
  sub_C46B40(&v12, a3);
  v7 = v13;
  v13 = 0;
  v15 = v7;
  v14 = v12;
  sub_AADC30(a1, (__int64)&v14, &v18);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return a1;
}
