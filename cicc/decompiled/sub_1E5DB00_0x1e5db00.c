// Function: sub_1E5DB00
// Address: 0x1e5db00
//
__int64 __fastcall sub_1E5DB00(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdi
  void (*v8)(); // rax
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned int v12; // ebx
  unsigned __int64 i; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // r8d
  int v18; // r9d
  _QWORD v20[6]; // [rsp+0h] [rbp-9B0h] BYREF
  _BYTE v21[296]; // [rsp+30h] [rbp-980h] BYREF
  _BYTE v22[1760]; // [rsp+158h] [rbp-858h] BYREF
  _QWORD *v23; // [rsp+838h] [rbp-178h]
  int v24; // [rsp+840h] [rbp-170h]
  char v25; // [rsp+844h] [rbp-16Ch]
  __int64 v26; // [rsp+848h] [rbp-168h]
  __int64 v27; // [rsp+850h] [rbp-160h]
  _QWORD *v28; // [rsp+858h] [rbp-158h]
  _BYTE v29[88]; // [rsp+860h] [rbp-150h] BYREF
  __int64 v30; // [rsp+8B8h] [rbp-F8h]
  __int64 v31; // [rsp+8C0h] [rbp-F0h]
  __int64 v32; // [rsp+8C8h] [rbp-E8h]
  __int64 v33; // [rsp+8D0h] [rbp-E0h]
  __int64 v34; // [rsp+8D8h] [rbp-D8h]
  __int64 v35; // [rsp+8E0h] [rbp-D0h]
  __int64 v36; // [rsp+8E8h] [rbp-C8h]
  __int64 v37; // [rsp+8F0h] [rbp-C0h]
  __int64 v38; // [rsp+8F8h] [rbp-B8h]
  __int64 v39; // [rsp+900h] [rbp-B0h]
  __int64 v40; // [rsp+908h] [rbp-A8h]
  __int64 v41; // [rsp+910h] [rbp-A0h]
  __int64 v42; // [rsp+918h] [rbp-98h]
  int v43; // [rsp+920h] [rbp-90h]
  __int64 v44; // [rsp+928h] [rbp-88h]
  _BYTE *v45; // [rsp+930h] [rbp-80h]
  _BYTE *v46; // [rsp+938h] [rbp-78h]
  __int64 v47; // [rsp+940h] [rbp-70h]
  int v48; // [rsp+948h] [rbp-68h]
  _BYTE v49[32]; // [rsp+950h] [rbp-60h] BYREF
  _QWORD v50[8]; // [rsp+970h] [rbp-40h] BYREF

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_23:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FC450C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_23;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FC450C);
  sub_1F03E40(v20, a1[29], a1[30], 0);
  v23 = a1;
  v20[0] = off_49FC1F0;
  v28 = a1 + 34;
  v24 = 0;
  v25 = 0;
  v26 = a2;
  v27 = v5;
  sub_1F024C0(v29, v21, v22);
  v30 = 0;
  v45 = v49;
  v46 = v49;
  v6 = a1[29];
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v47 = 4;
  v48 = 0;
  memset(v50, 0, 24);
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(void (**)())(*(_QWORD *)v7 + 240LL);
  if ( v8 != nullsub_705 )
    ((void (__fastcall *)(__int64, _QWORD *))v8)(v7, v50);
  v9 = **(_QWORD **)(a2 + 32);
  v10 = v9 + 24;
  sub_1F03410(v20, v9);
  v11 = *(_QWORD *)(v9 + 32);
  if ( v9 + 24 == v11 )
  {
    v12 = 0;
  }
  else
  {
    v12 = 0;
    do
    {
      v11 = *(_QWORD *)(v11 + 8);
      ++v12;
    }
    while ( v10 != v11 );
  }
  for ( i = sub_1DD5EE0(v9); v10 != i; --v12 )
  {
    while ( 1 )
    {
      if ( !i )
        BUG();
      if ( (*(_BYTE *)i & 4) == 0 )
        break;
      i = *(_QWORD *)(i + 8);
      --v12;
      if ( v10 == i )
        goto LABEL_16;
    }
    while ( (*(_BYTE *)(i + 46) & 8) != 0 )
      i = *(_QWORD *)(i + 8);
    i = *(_QWORD *)(i + 8);
  }
LABEL_16:
  v14 = sub_1DD5EE0(v9);
  sub_1F03430(v20, v9, *(_QWORD *)(v9 + 32), v14, v12);
  sub_1E5B110((__int64)v20);
  nullsub_753(v20);
  sub_1E41F20((__int64)v20, v9, v15, v16, v17, v18);
  return sub_1E43350(v20);
}
