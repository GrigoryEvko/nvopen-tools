// Function: sub_2737780
// Address: 0x2737780
//
__int64 __fastcall sub_2737780(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rbx
  __int64 *i; // r13
  __int64 v16; // rdx
  unsigned int v17; // r14d
  __int64 v18; // rbx
  __int64 j; // r13
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9

  *(_QWORD *)(a1 + 16) = a5;
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 32) = sub_B2BEC0(a2);
  v9 = sub_B2BE50(a2);
  *(_QWORD *)(a1 + 40) = a6;
  *(_QWORD *)(a1 + 24) = v9;
  *(_QWORD *)(a1 + 48) = a7;
  *(_BYTE *)(a1 + 56) = sub_11F2A60(*(_QWORD *)(a6 + 72), a7, a5);
  sub_2736860((__int64 *)a1, a2);
  if ( *(_QWORD *)(a1 + 64) != *(_QWORD *)(a1 + 72) )
  {
    a2 = 0;
    sub_2735F90((__int64 *)a1, 0, v10);
  }
  v14 = *(__int64 **)(a1 + 120);
  for ( i = &v14[4 * *(unsigned int *)(a1 + 128)]; i != v14; v14 += 4 )
  {
    if ( v14[2] != v14[1] )
    {
      a2 = *v14;
      sub_2735F90((__int64 *)a1, *v14, v10);
    }
  }
  v16 = *(unsigned int *)(a1 + 144);
  v17 = 0;
  if ( (_DWORD)v16 )
  {
    a2 = 0;
    v17 = sub_2737010((_QWORD *)a1, 0, v16, v11, v12, v13);
  }
  v18 = *(_QWORD *)(a1 + 5560);
  for ( j = v18 + 5400LL * *(unsigned int *)(a1 + 5568); j != v18; v17 |= sub_2737010(
                                                                            (_QWORD *)a1,
                                                                            a2,
                                                                            v16,
                                                                            v11,
                                                                            v12,
                                                                            v13) )
  {
    while ( !*(_DWORD *)(v18 + 16) )
    {
      v18 += 5400;
      if ( j == v18 )
        goto LABEL_14;
    }
    a2 = *(_QWORD *)v18;
    v18 += 5400;
  }
LABEL_14:
  sub_2730CF0(a1);
  sub_27320E0(a1, a2, v20, v21, v22, v23);
  return v17;
}
