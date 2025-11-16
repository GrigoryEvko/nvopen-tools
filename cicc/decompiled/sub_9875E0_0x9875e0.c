// Function: sub_9875E0
// Address: 0x9875e0
//
__int64 __fastcall sub_9875E0(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned int v4; // r14d
  __int64 v5; // rax
  unsigned int v6; // edx
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-38h]
  __int64 v12; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-28h]

  v4 = *((_DWORD *)a2 + 2);
  if ( v4 <= 0x40 )
  {
    v5 = *a3;
    if ( *a2 != *a3 )
      goto LABEL_4;
LABEL_11:
    sub_AADB10(a1, v4, 1);
    return a1;
  }
  if ( (unsigned __int8)sub_C43C50(a2, a3) )
    goto LABEL_11;
  v5 = *a3;
LABEL_4:
  v6 = *((_DWORD *)a3 + 2);
  v12 = v5;
  *((_DWORD *)a3 + 2) = 0;
  v7 = *((_DWORD *)a2 + 2);
  *((_DWORD *)a2 + 2) = 0;
  v11 = v7;
  v8 = *a2;
  v13 = v6;
  v10 = v8;
  sub_AADC30(a1, &v10, &v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v13 > 0x40 && v12 )
  {
    j_j___libc_free_0_0(v12);
    return a1;
  }
  return a1;
}
