// Function: sub_AC00B0
// Address: 0xac00b0
//
char *__fastcall sub_AC00B0(__int64 *a1, __int64 *a2, __int64 a3)
{
  char *result; // rax
  __int64 v5; // r14
  __int64 v6; // rdx
  int v7; // esi
  __int64 v8; // rcx
  __int64 v9; // rdx
  int v10; // ecx
  int v11; // ecx
  __int64 v12; // r12
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-58h]
  __int64 v19; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-48h]
  __int64 v21; // [rsp+30h] [rbp-40h]
  unsigned int v22; // [rsp+38h] [rbp-38h]

  result = (char *)sub_C4C880(a2, a3);
  if ( (int)result >= 0 )
    return result;
  v5 = *a1;
  v18 = *(_DWORD *)(a3 + 8);
  if ( v18 > 0x40 )
    sub_C43780(&v17, a3);
  else
    v17 = *(_QWORD *)a3;
  v16 = *((_DWORD *)a2 + 2);
  if ( v16 > 0x40 )
    sub_C43780(&v15, a2);
  else
    v15 = *a2;
  sub_AADC30((__int64)&v19, (__int64)&v15, &v17);
  v6 = *(unsigned int *)(v5 + 8);
  v7 = *(_DWORD *)(v5 + 8);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 12) )
  {
    v12 = *(_QWORD *)v5;
    if ( *(_QWORD *)v5 > (unsigned __int64)&v19 )
    {
      v13 = v6 + 1;
      v14 = v5;
    }
    else
    {
      v13 = v6 + 1;
      v14 = v5;
      if ( (unsigned __int64)&v19 < v12 + 32 * v6 )
      {
        sub_9D5330(v5, v6 + 1);
        v8 = *(_QWORD *)v5;
        v6 = *(unsigned int *)(v5 + 8);
        result = (char *)&v19 + *(_QWORD *)v5 - v12;
        v7 = *(_DWORD *)(v5 + 8);
        goto LABEL_9;
      }
    }
    sub_9D5330(v14, v13);
    v6 = *(unsigned int *)(v5 + 8);
    v8 = *(_QWORD *)v5;
    result = (char *)&v19;
    v7 = *(_DWORD *)(v5 + 8);
    goto LABEL_9;
  }
  v8 = *(_QWORD *)v5;
  result = (char *)&v19;
LABEL_9:
  v9 = v8 + 32 * v6;
  if ( v9 )
  {
    v10 = *((_DWORD *)result + 2);
    *((_DWORD *)result + 2) = 0;
    *(_DWORD *)(v9 + 8) = v10;
    *(_QWORD *)v9 = *(_QWORD *)result;
    v11 = *((_DWORD *)result + 6);
    *((_DWORD *)result + 6) = 0;
    *(_DWORD *)(v9 + 24) = v11;
    *(_QWORD *)(v9 + 16) = *((_QWORD *)result + 2);
    v7 = *(_DWORD *)(v5 + 8);
  }
  *(_DWORD *)(v5 + 8) = v7 + 1;
  if ( v22 > 0x40 && v21 )
    result = (char *)j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    result = (char *)j_j___libc_free_0_0(v19);
  if ( v16 > 0x40 && v15 )
    result = (char *)j_j___libc_free_0_0(v15);
  if ( v18 > 0x40 )
  {
    if ( v17 )
      return (char *)j_j___libc_free_0_0(v17);
  }
  return result;
}
