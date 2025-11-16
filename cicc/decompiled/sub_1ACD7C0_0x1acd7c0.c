// Function: sub_1ACD7C0
// Address: 0x1acd7c0
//
__int64 __fastcall sub_1ACD7C0(__int64 *a1)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  const void *v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdi
  const void *v10; // r12
  unsigned __int64 v11; // r8
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  const void *v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 i; // r15
  __int64 v22; // rsi
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // [rsp+8h] [rbp-38h]

  v2 = sub_1ACAC80((__int64)a1, *(_QWORD *)(*a1 + 112), *(_QWORD *)(a1[1] + 112));
  if ( v2 )
    return v2;
  v2 = sub_1ACA9E0((__int64)a1, (*(_WORD *)(*a1 + 18) & 0x4000) != 0, (*(_WORD *)(a1[1] + 18) & 0x4000) != 0);
  if ( v2 )
    return v2;
  v4 = *a1;
  if ( (*(_BYTE *)(*a1 + 19) & 0x40) != 0 )
  {
    v5 = sub_15E0FA0(a1[1]);
    v6 = *(_QWORD *)(v5 + 8);
    v7 = *(const void **)v5;
    v8 = sub_15E0FA0(*a1);
    v2 = sub_1ACABE0((__int64)a1, *(const void **)v8, *(_QWORD *)(v8 + 8), v7, v6);
    if ( v2 )
      return v2;
    v4 = *a1;
  }
  v2 = sub_1ACA9E0((__int64)a1, (*(_DWORD *)(v4 + 32) >> 21) & 1, (*(_DWORD *)(a1[1] + 32) >> 21) & 1);
  if ( v2 )
    return v2;
  v9 = *a1;
  if ( (*(_BYTE *)(*a1 + 34) & 0x20) == 0 )
  {
LABEL_13:
    v2 = sub_1ACA9E0(
           (__int64)a1,
           *(_DWORD *)(*(_QWORD *)(v9 + 24) + 8LL) >> 8 != 0,
           *(_DWORD *)(*(_QWORD *)(a1[1] + 24) + 8LL) >> 8 != 0);
    if ( v2 )
      return v2;
    v2 = sub_1ACA9E0((__int64)a1, (*(_WORD *)(*a1 + 18) >> 4) & 0x3FF, (*(_WORD *)(a1[1] + 18) >> 4) & 0x3FF);
    if ( v2 )
      return v2;
    v15 = *(_QWORD *)(*a1 + 24);
    v2 = sub_1ACB220(a1, v15, *(_QWORD *)(a1[1] + 24));
    if ( v2 )
      return v2;
    v16 = *a1;
    if ( (*(_BYTE *)(*a1 + 18) & 1) != 0 )
    {
      sub_15E08E0(*a1, v15);
      v17 = a1[1];
      v18 = *(_QWORD *)(v16 + 88);
      if ( (*(_BYTE *)(v17 + 18) & 1) == 0 )
      {
LABEL_25:
        v16 = *a1;
        v19 = *(_QWORD *)(v17 + 88);
        if ( (*(_BYTE *)(*a1 + 18) & 1) != 0 )
          sub_15E08E0(*a1, v15);
        v20 = *(_QWORD *)(v16 + 88);
        goto LABEL_19;
      }
    }
    else
    {
      v17 = a1[1];
      v18 = *(_QWORD *)(v16 + 88);
      if ( (*(_BYTE *)(v17 + 18) & 1) == 0 )
      {
        v19 = *(_QWORD *)(v17 + 88);
        v20 = *(_QWORD *)(v16 + 88);
LABEL_19:
        for ( i = v20 + 40LL * *(_QWORD *)(v16 + 96); v18 != i; v19 += 40 )
        {
          v22 = v18;
          v18 += 40;
          sub_1ACCBA0((__int64)a1, v22, v19);
        }
        return v2;
      }
    }
    sub_15E08E0(v17, v15);
    goto LABEL_25;
  }
  v10 = 0;
  v11 = 0;
  if ( (*(_BYTE *)(a1[1] + 34) & 0x20) == 0
    || (v23 = sub_15E61A0(a1[1]), v9 = *a1, v10 = (const void *)v23, v11 = v24, (*(_BYTE *)(*a1 + 34) & 0x20) != 0) )
  {
    v25 = v11;
    v12 = sub_15E61A0(v9);
    v11 = v25;
    v14 = (const void *)v12;
  }
  else
  {
    v13 = 0;
    v14 = 0;
  }
  v2 = sub_1ACABE0((__int64)a1, v14, v13, v10, v11);
  if ( !v2 )
  {
    v9 = *a1;
    goto LABEL_13;
  }
  return v2;
}
