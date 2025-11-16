// Function: sub_D5EA50
// Address: 0xd5ea50
//
__int64 __fastcall sub_D5EA50(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  const void *v3; // rdx
  const void *v4; // rdx
  unsigned int v5; // eax
  unsigned int v7; // eax
  const void *v8; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-38h]
  const void *v10; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-28h]

  v2 = *(_DWORD *)(a2 + 48);
  v11 = v2;
  if ( v2 <= 0x40 )
  {
    v3 = *(const void **)(a2 + 40);
    v9 = v2;
    v10 = v3;
LABEL_3:
    v4 = *(const void **)(a2 + 40);
    *(_DWORD *)(a1 + 8) = v2;
    v8 = v4;
LABEL_4:
    *(_QWORD *)a1 = v8;
    goto LABEL_5;
  }
  sub_C43780((__int64)&v10, (const void **)(a2 + 40));
  v2 = *(_DWORD *)(a2 + 48);
  v9 = v2;
  if ( v2 <= 0x40 )
    goto LABEL_3;
  sub_C43780((__int64)&v8, (const void **)(a2 + 40));
  v7 = v9;
  *(_DWORD *)(a1 + 8) = v9;
  if ( v7 <= 0x40 )
    goto LABEL_4;
  sub_C43780(a1, &v8);
LABEL_5:
  v5 = v11;
  *(_DWORD *)(a1 + 24) = v11;
  if ( v5 > 0x40 )
    sub_C43780(a1 + 16, &v10);
  else
    *(_QWORD *)(a1 + 16) = v10;
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return a1;
}
