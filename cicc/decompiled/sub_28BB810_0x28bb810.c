// Function: sub_28BB810
// Address: 0x28bb810
//
__int64 __fastcall sub_28BB810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r15
  unsigned int v9; // esi
  __int64 v10; // r10
  int v11; // r9d
  __int64 *v12; // r8
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rcx
  int v16; // eax
  unsigned int v17; // eax
  int v18; // ecx
  int v19; // edx
  __int64 v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v22; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-48h]
  unsigned __int64 v25; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-38h]

  if ( *(_BYTE *)a2 != 61
    || (unsigned __int8)sub_B463C0(a2, *(_QWORD *)(a2 + 40))
    || sub_B46500((unsigned __int8 *)a2)
    || (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    goto LABEL_2;
  }
  v6 = *(_QWORD *)(a2 - 32);
  v7 = *(_QWORD *)(v6 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v7 + 16);
  if ( *(_DWORD *)(v7 + 8) >> 8 || (v8 = sub_B43CC0(a2), !sub_D30730(v6, *(_QWORD *)(a2 + 8), v8, 0, 0, 0, 0)) )
  {
LABEL_2:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    *(_DWORD *)(a1 + 32) = 1;
    return a1;
  }
  v24 = sub_AE43F0(v8, *(_QWORD *)(v6 + 8));
  if ( v24 > 0x40 )
    sub_C43690((__int64)&v23, 0, 0);
  else
    v23 = 0;
  v20 = 0;
  if ( *(_BYTE *)v6 == 63 )
  {
    if ( (unsigned __int8)sub_B463C0(v6, *(_QWORD *)(a2 + 40)) || !(unsigned __int8)sub_B4DE60(v6, v8, (__int64)&v23) )
    {
      *(_QWORD *)a1 = 0;
      *(_QWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 1;
      goto LABEL_19;
    }
    v20 = v6;
    v6 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
  }
  v26 = v24;
  if ( v24 > 0x40 )
    sub_C43780((__int64)&v25, (const void **)&v23);
  else
    v25 = v23;
  v9 = *(_DWORD *)(a3 + 32);
  v21 = v6;
  if ( !v9 )
  {
    ++*(_QWORD *)(a3 + 8);
    v22 = 0;
    goto LABEL_42;
  }
  v10 = *(_QWORD *)(a3 + 16);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v14 = (__int64 *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( v6 != *v14 )
  {
    while ( v15 != -4096 )
    {
      if ( !v12 && v15 == -8192 )
        v12 = v14;
      v13 = (v9 - 1) & (v11 + v13);
      v14 = (__int64 *)(v10 + 16LL * v13);
      v15 = *v14;
      if ( v6 == *v14 )
        goto LABEL_18;
      ++v11;
    }
    v18 = *(_DWORD *)(a3 + 24);
    if ( v12 )
      v14 = v12;
    ++*(_QWORD *)(a3 + 8);
    v19 = v18 + 1;
    v22 = v14;
    if ( 4 * (v18 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a3 + 28) - v19 > v9 >> 3 )
      {
LABEL_36:
        *(_DWORD *)(a3 + 24) = v19;
        if ( *v14 != -4096 )
          --*(_DWORD *)(a3 + 28);
        *v14 = v6;
        *((_DWORD *)v14 + 2) = (*(_DWORD *)a3)++;
        goto LABEL_18;
      }
LABEL_43:
      sub_28BB630(a3 + 8, v9);
      sub_28BB570(a3 + 8, &v21, &v22);
      v6 = v21;
      v19 = *(_DWORD *)(a3 + 24) + 1;
      v14 = v22;
      goto LABEL_36;
    }
LABEL_42:
    v9 *= 2;
    goto LABEL_43;
  }
LABEL_18:
  v16 = *((_DWORD *)v14 + 2);
  *(_QWORD *)(a1 + 8) = a2;
  *(_DWORD *)(a1 + 16) = v16;
  v17 = v26;
  *(_QWORD *)a1 = v20;
  *(_DWORD *)(a1 + 32) = v17;
  *(_QWORD *)(a1 + 24) = v25;
LABEL_19:
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  return a1;
}
