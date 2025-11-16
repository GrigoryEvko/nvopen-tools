// Function: sub_144B1E0
// Address: 0x144b1e0
//
__int64 __fastcall sub_144B1E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  bool v4; // zf
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rbx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  v3 = a2[1];
  if ( v3 != *a2 )
  {
    while ( (*(int (__fastcall **)(_QWORD))(**(_QWORD **)(v3 - 8) + 40LL))(*(_QWORD *)(v3 - 8)) > 5 )
    {
      sub_160FB80(a2);
      v3 = a2[1];
      if ( *a2 == v3 )
        goto LABEL_6;
    }
    v3 = a2[1];
  }
LABEL_6:
  v4 = (*(unsigned int (__fastcall **)(_QWORD))(**(_QWORD **)(v3 - 8) + 40LL))(*(_QWORD *)(v3 - 8)) == 5;
  v5 = a2[1];
  if ( v4 )
  {
    v6 = *(_QWORD *)(v5 - 8);
    if ( !v6 )
      v6 = 160;
    return sub_1617B20(v6, a1, 1);
  }
  v8 = *(_QWORD *)(v5 - 8);
  v9 = sub_22077B0(672);
  v10 = v9;
  if ( !v9 )
  {
    v11 = a2[1];
    v12 = *a2;
    v17 = 160;
    if ( v11 == *a2 )
    {
      v15 = *(_QWORD *)(v8 + 16);
      v16 = 0;
      goto LABEL_15;
    }
    goto LABEL_12;
  }
  sub_144AF60(v9);
  v11 = a2[1];
  v12 = *a2;
  if ( v11 != *a2 )
  {
LABEL_12:
    v13 = v10 + 328;
    do
    {
      v14 = *(_QWORD *)(v11 - 8);
      v11 -= 8;
      v13 += 8;
      *(_QWORD *)(v13 - 8) = v14 + 224;
    }
    while ( v11 != v12 );
  }
  v15 = *(_QWORD *)(v8 + 16);
  v16 = v10 + 160;
  v17 = v10 + 160;
LABEL_15:
  v18 = *(unsigned int *)(v15 + 120);
  if ( (unsigned int)v18 >= *(_DWORD *)(v15 + 124) )
  {
    v20 = v17;
    sub_16CD150(v15 + 112, v15 + 128, 0, 8);
    v18 = *(unsigned int *)(v15 + 120);
    v17 = v20;
  }
  v19 = v17;
  *(_QWORD *)(*(_QWORD *)(v15 + 112) + 8 * v18) = v16;
  ++*(_DWORD *)(v15 + 120);
  sub_16185C0(v15, v10);
  sub_16110B0(a2, v16);
  v6 = v19;
  return sub_1617B20(v6, a1, 1);
}
