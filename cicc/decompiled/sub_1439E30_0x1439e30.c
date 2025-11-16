// Function: sub_1439E30
// Address: 0x1439e30
//
__int64 __fastcall sub_1439E30(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 v3; // rax
  _QWORD *v4; // rax
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // [rsp+8h] [rbp-38h]

  v2 = 0;
  v3 = sub_15E0530(a2);
  if ( (unsigned __int8)sub_1602770(v3) )
  {
    v8 = *(__int64 **)(a1 + 8);
    v9 = *v8;
    v10 = v8[1];
    if ( v9 == v10 )
LABEL_17:
      BUG();
    while ( *(_UNKNOWN **)v9 != &unk_4F99115 )
    {
      v9 += 16;
      if ( v10 == v9 )
        goto LABEL_17;
    }
    v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F99115);
    v12 = v11;
    v2 = (__int64 *)(v11 + 160);
    if ( !*(_BYTE *)(v11 + 168) )
    {
      v13 = *(_QWORD *)(v11 + 192);
      v14 = *(_QWORD *)(v11 + 184);
      v15 = *(_QWORD *)(v14 + 160);
      if ( !*(_BYTE *)(v15 + 408) )
      {
        v16 = v13;
        sub_137CAE0(*(_QWORD *)(v14 + 160), *(__int64 **)(v15 + 416), *(_QWORD *)(v15 + 424), *(_QWORD **)(v15 + 432));
        *(_BYTE *)(v15 + 408) = 1;
        v13 = v16;
      }
      sub_1370060(v2, *(const void **)(v12 + 176), v15, v13);
      *(_BYTE *)(v12 + 168) = 1;
    }
  }
  v4 = (_QWORD *)sub_22077B0(24);
  if ( v4 )
  {
    *v4 = a2;
    v4[1] = v2;
    v4[2] = 0;
  }
  v5 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v4;
  if ( v5 )
  {
    v6 = *(_QWORD *)(v5 + 16);
    if ( v6 )
    {
      sub_1368A00(*(__int64 **)(v5 + 16));
      j_j___libc_free_0(v6, 8);
    }
    j_j___libc_free_0(v5, 24);
  }
  return 0;
}
