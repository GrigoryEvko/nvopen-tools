// Function: sub_29DB0A0
// Address: 0x29db0a0
//
__int64 __fastcall sub_29DB0A0(__int64 *a1)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  const void *v7; // r13
  __int64 v8; // rax
  _BOOL8 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdi
  const void *v12; // r12
  unsigned __int64 v13; // r8
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  const void *v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // r14
  __int64 v24; // rdx
  __int64 i; // r15
  __int64 v26; // rax
  unsigned __int64 v27; // [rsp+8h] [rbp-38h]

  v2 = sub_29D84C0(a1, *(_QWORD *)(*a1 + 120), *(_QWORD *)(a1[1] + 120));
  if ( v2 )
    return v2;
  v2 = sub_29D7CF0((__int64)a1, (*(_WORD *)(*a1 + 2) & 0x4000) != 0, (*(_WORD *)(a1[1] + 2) & 0x4000) != 0);
  if ( v2 )
    return v2;
  v4 = *a1;
  if ( (*(_BYTE *)(*a1 + 3) & 0x40) != 0 )
  {
    v5 = sub_B2DBE0(a1[1]);
    v6 = *(_QWORD *)(v5 + 8);
    v7 = *(const void **)v5;
    v8 = sub_B2DBE0(*a1);
    v2 = sub_29D7F50((__int64)a1, *(const void **)v8, *(_QWORD *)(v8 + 8), v7, v6);
    if ( v2 )
      return v2;
    v4 = *a1;
  }
  v9 = (*(_WORD *)(v4 + 34) & 0x400) != 0;
  v2 = sub_29D7CF0((__int64)a1, v9, (*(_WORD *)(a1[1] + 34) & 0x400) != 0);
  if ( v2 )
    return v2;
  v11 = *a1;
  if ( (*(_BYTE *)(*a1 + 35) & 4) != 0 )
  {
    v12 = 0;
    v13 = 0;
    if ( (*(_BYTE *)(a1[1] + 35) & 4) == 0
      || (v26 = sub_B31D10(a1[1], v9, v10),
          v11 = *a1,
          v12 = (const void *)v26,
          v13 = v10,
          (*(_BYTE *)(*a1 + 35) & 4) != 0) )
    {
      v27 = v13;
      v14 = sub_B31D10(v11, v9, v10);
      v13 = v27;
      v16 = (const void *)v14;
    }
    else
    {
      v15 = 0;
      v16 = 0;
    }
    v2 = sub_29D7F50((__int64)a1, v16, v15, v12, v13);
    if ( v2 )
      return v2;
    v11 = *a1;
  }
  v2 = sub_29D7CF0(
         (__int64)a1,
         *(_DWORD *)(*(_QWORD *)(v11 + 24) + 8LL) >> 8 != 0,
         *(_DWORD *)(*(_QWORD *)(a1[1] + 24) + 8LL) >> 8 != 0);
  if ( v2 )
    return v2;
  v2 = sub_29D7CF0((__int64)a1, (*(_WORD *)(*a1 + 2) >> 4) & 0x3FF, (*(_WORD *)(a1[1] + 2) >> 4) & 0x3FF);
  if ( v2 )
    return v2;
  v17 = *(_QWORD *)(*a1 + 24);
  v2 = sub_29D81B0(a1, v17, *(_QWORD *)(a1[1] + 24));
  if ( v2 )
    return v2;
  v20 = *a1;
  if ( (*(_BYTE *)(*a1 + 2) & 1) != 0 )
  {
    sub_B2C6D0(*a1, v17, v18, v19);
    v21 = a1[1];
    v22 = *(_QWORD *)(v20 + 96);
    if ( (*(_BYTE *)(v21 + 2) & 1) == 0 )
    {
LABEL_27:
      v20 = *a1;
      v23 = *(_QWORD *)(v21 + 96);
      if ( (*(_BYTE *)(*a1 + 2) & 1) != 0 )
        sub_B2C6D0(*a1, v17, v18, v19);
      v24 = *(_QWORD *)(v20 + 96);
      goto LABEL_19;
    }
LABEL_30:
    sub_B2C6D0(v21, v17, v18, v19);
    goto LABEL_27;
  }
  v21 = a1[1];
  v22 = *(_QWORD *)(v20 + 96);
  if ( (*(_BYTE *)(v21 + 2) & 1) != 0 )
    goto LABEL_30;
  v23 = *(_QWORD *)(v21 + 96);
  v24 = *(_QWORD *)(v20 + 96);
LABEL_19:
  for ( i = v24 + 40LL * *(_QWORD *)(v20 + 104); v22 != i; v23 += 40 )
  {
    if ( (unsigned int)sub_29DA390((__int64)a1, v22, v23) )
      BUG();
    v22 += 40;
  }
  return v2;
}
