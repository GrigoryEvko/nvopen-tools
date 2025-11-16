// Function: sub_E26960
// Address: 0xe26960
//
unsigned __int64 __fastcall sub_E26960(__int64 a1, __int64 **a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rcx
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r15
  __int64 v10; // rdx
  unsigned __int64 v11; // r13
  unsigned __int64 *v13; // rax
  __int64 *v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // r13
  __int64 v17; // rax
  __int64 *v18; // rax
  unsigned __int64 *v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  v7 = **a2;
  v8 = (v7 + (*a2)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL;
  (*a2)[1] = v8 - v7 + 40;
  if ( (*a2)[1] <= (unsigned __int64)(*a2)[2] )
  {
    if ( v8 )
    {
      *(_DWORD *)(v8 + 8) = 5;
      *(_QWORD *)(v8 + 16) = 0;
      *(_QWORD *)(v8 + 24) = 0;
      *(_QWORD *)v8 = &unk_49E0F88;
      *(_QWORD *)(v8 + 32) = 0;
      goto LABEL_4;
    }
LABEL_17:
    MEMORY[0x18] = 0;
    BUG();
  }
  v20 = a5;
  v15 = (__int64 *)sub_22077B0(32);
  v16 = v15;
  if ( v15 )
  {
    *v15 = 0;
    v15[1] = 0;
    v15[2] = 0;
    v15[3] = 0;
  }
  v17 = sub_2207820(4096);
  v16[2] = 4096;
  *v16 = v17;
  v8 = v17;
  v18 = *a2;
  v16[1] = 40;
  v16[3] = (__int64)v18;
  *a2 = v16;
  if ( !v8 )
    goto LABEL_17;
  *(_DWORD *)(v8 + 8) = 5;
  *(_QWORD *)(v8 + 16) = 0;
  a5 = v20;
  *(_QWORD *)(v8 + 24) = 0;
  *(_QWORD *)v8 = &unk_49E0F88;
  *(_QWORD *)(v8 + 32) = 0;
LABEL_4:
  *(_QWORD *)(v8 + 24) = a4;
  *(_QWORD *)(v8 + 32) = a5;
  v9 = sub_E263F0(a1, a3, v8);
  v10 = **a2;
  v11 = (v10 + (*a2)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL;
  (*a2)[1] = v11 - v10 + 40;
  if ( (*a2)[1] > (unsigned __int64)(*a2)[2] )
  {
    v13 = (unsigned __int64 *)sub_22077B0(32);
    if ( v13 )
    {
      *v13 = 0;
      v13[1] = 0;
      v13[2] = 0;
      v13[3] = 0;
    }
    v19 = v13;
    v11 = sub_2207820(4096);
    *v19 = v11;
    v14 = *a2;
    v19[2] = 4096;
    v19[3] = (unsigned __int64)v14;
    *a2 = (__int64 *)v19;
    v19[1] = 40;
  }
  if ( !v11 )
  {
    MEMORY[0x10] = v9;
    BUG();
  }
  *(_QWORD *)(v11 + 16) = 0;
  *(_DWORD *)(v11 + 8) = 27;
  *(_BYTE *)(v11 + 24) = 0;
  *(_QWORD *)v11 = &unk_49E11E0;
  *(_QWORD *)(v11 + 32) = 0;
  *(_QWORD *)(v11 + 16) = v9;
  if ( !(unsigned __int8)sub_E20730((size_t *)a3, 1u, "8") )
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  return v11;
}
