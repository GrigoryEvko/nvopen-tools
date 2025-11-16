// Function: sub_2906B20
// Address: 0x2906b20
//
unsigned __int64 __fastcall sub_2906B20(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v6; // rax
  unsigned int v8; // esi
  int v9; // eax
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rdi
  int v13; // eax
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 *v20; // rax
  unsigned __int64 v21; // rdi
  int v22; // r12d
  int v23; // eax
  __int64 v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25; // [rsp+18h] [rbp-48h] BYREF
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  int v27; // [rsp+28h] [rbp-38h]

  v6 = *a2;
  v27 = 0;
  v26 = v6;
  if ( (unsigned __int8)sub_22B1A50(a1, &v26, &v24) )
    return *(_QWORD *)(a1 + 32) + ((unsigned __int64)*(unsigned int *)(v24 + 8) << 6);
  v8 = *(_DWORD *)(a1 + 24);
  v9 = *(_DWORD *)(a1 + 16);
  v10 = v24;
  ++*(_QWORD *)a1;
  v11 = v9 + 1;
  v25 = v10;
  if ( 4 * v11 >= 3 * v8 )
  {
    v8 *= 2;
    goto LABEL_13;
  }
  if ( v8 - *(_DWORD *)(a1 + 20) - v11 <= v8 >> 3 )
  {
LABEL_13:
    sub_D39D40(a1, v8);
    sub_22B1A50(a1, &v26, &v25);
    v10 = v25;
    v11 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v11;
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v10 = v26;
  *(_DWORD *)(v10 + 8) = v27;
  *(_DWORD *)(v10 + 8) = *(_DWORD *)(a1 + 40);
  v12 = *(unsigned int *)(a1 + 40);
  v13 = v12;
  if ( *(_DWORD *)(a1 + 44) <= (unsigned int)v12 )
  {
    v18 = sub_C8D7D0(a1 + 32, a1 + 48, 0, 0x40u, (unsigned __int64 *)&v26, a1 + 32);
    v19 = a1 + 32;
    v14 = v18;
    v20 = (__int64 *)(v18 + ((unsigned __int64)*(unsigned int *)(a1 + 40) << 6));
    if ( v20 )
    {
      *v20 = *a2;
      sub_28FF950((__int64)(v20 + 1), a3);
      v19 = a1 + 32;
    }
    sub_28FFEE0(v19, v14);
    v21 = *(_QWORD *)(a1 + 32);
    v22 = v26;
    if ( a1 + 48 != v21 )
      _libc_free(v21);
    v23 = *(_DWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v14;
    *(_DWORD *)(a1 + 44) = v22;
    v17 = (unsigned int)(v23 + 1);
    *(_DWORD *)(a1 + 40) = v17;
  }
  else
  {
    v14 = *(_QWORD *)(a1 + 32);
    v15 = v14 + (v12 << 6);
    if ( v15 )
    {
      v16 = v15 + 8;
      *(_QWORD *)(v16 - 8) = *a2;
      sub_28FF950(v16, a3);
      v13 = *(_DWORD *)(a1 + 40);
      v14 = *(_QWORD *)(a1 + 32);
    }
    v17 = (unsigned int)(v13 + 1);
    *(_DWORD *)(a1 + 40) = v17;
  }
  return v14 + (v17 << 6) - 64;
}
