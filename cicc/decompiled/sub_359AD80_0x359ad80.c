// Function: sub_359AD80
// Address: 0x359ad80
//
void __fastcall sub_359AD80(__int64 *a1)
{
  __int64 v1; // rax
  __int64 *v2; // r13
  _QWORD *v4; // r15
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r9
  signed __int64 v15; // rsi
  __int64 v16; // rdi
  const char *v17; // rdx
  const char *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r12
  _DWORD *v21; // rdx
  __int64 v22; // rax
  int v23; // edx
  int v24; // r10d
  __int64 *v25; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v26[3]; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE v27[24]; // [rsp+38h] [rbp-B8h] BYREF
  const char *v28[4]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v29; // [rsp+70h] [rbp-80h]
  _QWORD v30[3]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v31; // [rsp+98h] [rbp-58h]
  _DWORD *v32; // [rsp+A0h] [rbp-50h]
  __int64 v33; // [rsp+A8h] [rbp-48h]
  unsigned __int64 *v34; // [rsp+B0h] [rbp-40h]

  v1 = a1[1];
  v2 = *(__int64 **)(v1 + 8);
  v25 = *(__int64 **)(v1 + 16);
  while ( v25 != v2 )
  {
    v20 = *v2;
    v26[1] = 0;
    v26[0] = (unsigned __int64)v27;
    v33 = 0x100000000LL;
    v26[2] = 16;
    v30[1] = 2;
    v30[0] = &unk_49DD288;
    v30[2] = 0;
    v34 = v26;
    v31 = 0;
    v32 = 0;
    sub_CB5980((__int64)v30, 0, 0, 0);
    v21 = v32;
    if ( (unsigned __int64)(v31 - (_QWORD)v32) > 5 )
    {
      *v32 = 1734440019;
      v4 = v30;
      *((_WORD *)v21 + 2) = 11621;
      v32 = (_DWORD *)((char *)v32 + 6);
    }
    else
    {
      v4 = (_QWORD *)sub_CB6200((__int64)v30, "Stage-", 6u);
    }
    v5 = sub_3598DB0(a1[1], v20);
    v6 = sub_CB59F0((__int64)v4, v5);
    v7 = *(_QWORD *)(v6 + 32);
    v8 = v6;
    if ( (unsigned __int64)(*(_QWORD *)(v6 + 24) - v7) <= 6 )
    {
      v8 = sub_CB6200(v6, "_Cycle-", 7u);
      v22 = a1[1];
      v10 = *(_QWORD *)(v22 + 40);
      v11 = *(unsigned int *)(v22 + 56);
      if ( !(_DWORD)v11 )
        goto LABEL_15;
    }
    else
    {
      *(_DWORD *)v7 = 1668891487;
      *(_WORD *)(v7 + 4) = 25964;
      *(_BYTE *)(v7 + 6) = 45;
      *(_QWORD *)(v6 + 32) += 7LL;
      v9 = a1[1];
      v10 = *(_QWORD *)(v9 + 40);
      v11 = *(unsigned int *)(v9 + 56);
      if ( !(_DWORD)v11 )
        goto LABEL_15;
    }
    v12 = (v11 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v20 != *v13 )
    {
      v23 = 1;
      while ( v14 != -4096 )
      {
        v24 = v23 + 1;
        v12 = (v11 - 1) & (v23 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( v20 == *v13 )
          goto LABEL_7;
        v23 = v24;
      }
LABEL_15:
      v15 = -1;
      goto LABEL_9;
    }
LABEL_7:
    if ( v13 == (__int64 *)(v10 + 16 * v11) )
      goto LABEL_15;
    v15 = *((int *)v13 + 2);
LABEL_9:
    sub_CB59F0(v8, v15);
    v16 = *(_QWORD *)(*a1 + 24);
    v17 = (const char *)v34[1];
    v18 = (const char *)*v34;
    v29 = 261;
    v28[1] = v17;
    v28[0] = v18;
    v19 = sub_E6C460(v16, v28);
    sub_2E87EC0(v20, *a1, v19);
    v30[0] = &unk_49DD388;
    sub_CB5840((__int64)v30);
    if ( (_BYTE *)v26[0] != v27 )
      _libc_free(v26[0]);
    ++v2;
  }
}
