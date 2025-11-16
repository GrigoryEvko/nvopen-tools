// Function: sub_C4CAA0
// Address: 0xc4caa0
//
__int64 __fastcall sub_C4CAA0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned int v7; // r15d
  unsigned int v8; // ecx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // esi
  bool v12; // zf
  __int64 v13; // rdx
  unsigned int v14; // ecx
  const void *v15; // rax
  unsigned int v16; // eax
  unsigned int v17; // esi
  bool v18; // zf
  __int64 v19; // rdx
  const void *v20; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-58h]
  __int64 v22; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-48h]
  const void *v24; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+38h] [rbp-38h]

  if ( a4 == 1 )
  {
    sub_C4A3E0(a1, a2, a3);
    return a1;
  }
  if ( (a4 & 0xFFFFFFFD) != 0 )
    BUG();
  v21 = 1;
  v20 = 0;
  v23 = 1;
  v22 = 0;
  sub_C4C400(a2, a3, (__int64)&v20, (__int64)&v22);
  v7 = v23;
  if ( v23 <= 0x40 )
  {
    v10 = v22;
    if ( v22 )
    {
      v9 = 1LL << ((unsigned __int8)v23 - 1);
      if ( a4 )
        goto LABEL_9;
      goto LABEL_22;
    }
LABEL_16:
    v16 = v21;
    v21 = 0;
    *(_DWORD *)(a1 + 8) = v16;
    *(_QWORD *)a1 = v20;
    goto LABEL_17;
  }
  if ( v7 == (unsigned int)sub_C444A0((__int64)&v22) )
    goto LABEL_16;
  v8 = v7 - 1;
  v9 = 1LL << ((unsigned __int8)v7 - 1);
  if ( a4 )
  {
    v10 = *(_QWORD *)(v22 + 8LL * (v8 >> 6));
LABEL_9:
    v11 = *(_DWORD *)(a3 + 8);
    v12 = (v9 & v10) == 0;
    v13 = *(_QWORD *)a3;
    if ( v11 > 0x40 )
      v13 = *(_QWORD *)(v13 + 8LL * ((v11 - 1) >> 6));
    v14 = v21;
    if ( ((v13 & (1LL << ((unsigned __int8)v11 - 1))) != 0) != !v12 )
      goto LABEL_12;
    v25 = v21;
    if ( v21 > 0x40 )
      sub_C43780((__int64)&v24, &v20);
    else
      v24 = v20;
    sub_C46A40((__int64)&v24, 1);
    goto LABEL_28;
  }
  v10 = *(_QWORD *)(v22 + 8LL * (v8 >> 6));
LABEL_22:
  v17 = *(_DWORD *)(a3 + 8);
  v18 = (v9 & v10) == 0;
  v19 = *(_QWORD *)a3;
  if ( v17 > 0x40 )
    v19 = *(_QWORD *)(v19 + 8LL * ((v17 - 1) >> 6));
  v14 = v21;
  if ( ((v19 & (1LL << ((unsigned __int8)v17 - 1))) != 0) != !v18 )
  {
    v25 = v21;
    if ( v21 > 0x40 )
      sub_C43780((__int64)&v24, &v20);
    else
      v24 = v20;
    sub_C46F20((__int64)&v24, 1u);
LABEL_28:
    v7 = v23;
    *(_DWORD *)(a1 + 8) = v25;
    *(_QWORD *)a1 = v24;
LABEL_17:
    if ( v7 <= 0x40 )
      goto LABEL_13;
LABEL_18:
    if ( v22 )
      j_j___libc_free_0_0(v22);
    goto LABEL_13;
  }
LABEL_12:
  v15 = v20;
  *(_DWORD *)(a1 + 8) = v14;
  v21 = 0;
  *(_QWORD *)a1 = v15;
  if ( v7 > 0x40 )
    goto LABEL_18;
LABEL_13:
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  return a1;
}
