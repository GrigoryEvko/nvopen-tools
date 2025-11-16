// Function: sub_D5DFC0
// Address: 0xd5dfc0
//
__int64 __fastcall sub_D5DFC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rdx
  unsigned int v6; // edx
  __int64 v7; // rax
  unsigned int v9; // eax
  const void *v10; // rdx
  unsigned int v11; // eax
  unsigned int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-68h] BYREF
  const void *v14; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-58h]
  unsigned __int64 v16; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-48h]
  unsigned __int64 v18; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+38h] [rbp-38h]
  char v20; // [rsp+40h] [rbp-30h]

  v5 = *(__int64 **)(a2 + 8);
  v13 = a2;
  sub_D5CDD0((__int64)&v18, a3, v5, sub_D5C780, (__int64)&v13);
  if ( v20 )
  {
    v6 = v19;
    v7 = 1LL << ((unsigned __int8)v19 - 1);
    if ( v19 > 0x40 )
    {
      if ( (*(_QWORD *)(v18 + 8LL * ((v19 - 1) >> 6)) & v7) != 0 )
        goto LABEL_4;
      v17 = v19;
      sub_C43780((__int64)&v16, (const void **)&v18);
    }
    else
    {
      if ( (v7 & v18) != 0 )
      {
LABEL_4:
        *(_QWORD *)a1 = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 8) = 1;
        *(_DWORD *)(a1 + 24) = 1;
        goto LABEL_5;
      }
      v17 = v19;
      v16 = v18;
    }
    v9 = *(_DWORD *)(a2 + 48);
    v15 = v9;
    if ( v9 > 0x40 )
    {
      sub_C43780((__int64)&v14, (const void **)(a2 + 40));
      v12 = v15;
      *(_DWORD *)(a1 + 8) = v15;
      if ( v12 > 0x40 )
      {
        sub_C43780(a1, &v14);
        goto LABEL_16;
      }
    }
    else
    {
      v10 = *(const void **)(a2 + 40);
      *(_DWORD *)(a1 + 8) = v9;
      v14 = v10;
    }
    *(_QWORD *)a1 = v14;
LABEL_16:
    v11 = v17;
    *(_DWORD *)(a1 + 24) = v17;
    if ( v11 > 0x40 )
      sub_C43780(a1 + 16, (const void **)&v16);
    else
      *(_QWORD *)(a1 + 16) = v16;
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( !v20 )
      return a1;
    v6 = v19;
LABEL_5:
    v20 = 0;
    if ( v6 > 0x40 )
    {
      if ( v18 )
        j_j___libc_free_0_0(v18);
    }
    return a1;
  }
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 8) = 1;
  *(_DWORD *)(a1 + 24) = 1;
  return a1;
}
