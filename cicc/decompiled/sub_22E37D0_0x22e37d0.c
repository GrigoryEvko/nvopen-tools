// Function: sub_22E37D0
// Address: 0x22e37d0
//
__int64 __fastcall sub_22E37D0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  bool v4; // zf
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 8);
  if ( v3 != *(_QWORD *)a2 )
  {
    while ( (*(int (__fastcall **)(_QWORD))(**(_QWORD **)(v3 - 8) + 40LL))(*(_QWORD *)(v3 - 8)) > 5 )
    {
      sub_B823C0(a2);
      v3 = *(_QWORD *)(a2 + 8);
      if ( *(_QWORD *)a2 == v3 )
        goto LABEL_6;
    }
    v3 = *(_QWORD *)(a2 + 8);
  }
LABEL_6:
  v4 = (*(unsigned int (__fastcall **)(_QWORD))(**(_QWORD **)(v3 - 8) + 40LL))(*(_QWORD *)(v3 - 8)) == 5;
  v5 = *(_QWORD *)(a2 + 8);
  if ( v4 )
  {
    v6 = *(_QWORD *)(v5 - 8);
    if ( !v6 )
      v6 = 176;
    return sub_B88F40(v6, a1, 1);
  }
  v8 = *(_QWORD *)(v5 - 8);
  v9 = sub_22077B0(0x298u);
  v11 = (__int64 *)v9;
  if ( !v9 )
  {
    v12 = *(_QWORD *)(a2 + 8);
    v13 = *(_QWORD *)a2;
    v18 = 176;
    if ( v12 == *(_QWORD *)a2 )
    {
      v16 = *(_QWORD *)(v8 + 8);
      v17 = 0;
      goto LABEL_15;
    }
    goto LABEL_12;
  }
  sub_22E3560(v9);
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(_QWORD *)a2;
  if ( v12 != *(_QWORD *)a2 )
  {
LABEL_12:
    v14 = v11 + 42;
    do
    {
      v15 = *(_QWORD *)(v12 - 8);
      v12 -= 8;
      *v14++ = v15 + 208;
    }
    while ( v12 != v13 );
  }
  v16 = *(_QWORD *)(v8 + 8);
  v17 = (__int64)(v11 + 22);
  v18 = (__int64)(v11 + 22);
LABEL_15:
  v19 = *(unsigned int *)(v16 + 120);
  if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 124) )
  {
    v21 = v18;
    sub_C8D5F0(v16 + 112, (const void *)(v16 + 128), v19 + 1, 8u, v18, v10);
    v19 = *(unsigned int *)(v16 + 120);
    v18 = v21;
  }
  v20 = v18;
  *(_QWORD *)(*(_QWORD *)(v16 + 112) + 8 * v19) = v17;
  ++*(_DWORD *)(v16 + 120);
  sub_B8B080(v16, v11);
  sub_B841D0((char **)a2, v17);
  v6 = v20;
  return sub_B88F40(v6, a1, 1);
}
