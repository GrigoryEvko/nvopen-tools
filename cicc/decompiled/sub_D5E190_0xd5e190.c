// Function: sub_D5E190
// Address: 0xd5e190
//
__int64 __fastcall sub_D5E190(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  const void *v5; // rdx
  const void *v6; // rdx
  unsigned int v7; // eax
  unsigned int v8; // eax
  const void *v9; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-38h]
  const void *v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]

  if ( *(_BYTE *)(a2 + 18) || *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL) >> 8 )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  v4 = *(_DWORD *)(a2 + 48);
  v12 = v4;
  if ( v4 <= 0x40 )
  {
    v5 = *(const void **)(a2 + 40);
    v10 = v4;
    v11 = v5;
LABEL_7:
    v6 = *(const void **)(a2 + 40);
    *(_DWORD *)(a1 + 8) = v4;
    v9 = v6;
    goto LABEL_8;
  }
  sub_C43780((__int64)&v11, (const void **)(a2 + 40));
  v4 = *(_DWORD *)(a2 + 48);
  v10 = v4;
  if ( v4 <= 0x40 )
    goto LABEL_7;
  sub_C43780((__int64)&v9, (const void **)(a2 + 40));
  v8 = v10;
  *(_DWORD *)(a1 + 8) = v10;
  if ( v8 > 0x40 )
  {
    sub_C43780(a1, &v9);
    goto LABEL_9;
  }
LABEL_8:
  *(_QWORD *)a1 = v9;
LABEL_9:
  v7 = v12;
  *(_DWORD *)(a1 + 24) = v12;
  if ( v7 > 0x40 )
    sub_C43780(a1 + 16, &v11);
  else
    *(_QWORD *)(a1 + 16) = v11;
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  return a1;
}
