// Function: sub_16B7D30
// Address: 0x16b7d30
//
void __fastcall sub_16B7D30(__int64 a1, __int64 a2, __int64 a3, const char *a4, size_t a5)
{
  __int64 v5; // rbx
  __int64 v9; // rdi
  unsigned int v12; // eax
  _QWORD *v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // r10d
  _QWORD *v20; // rcx
  _QWORD *v21; // r15
  void *v22; // rdi
  _QWORD *v23; // rbx
  _QWORD *v24; // r15
  _QWORD *v25; // rax
  __int64 v26; // rax
  void *v27; // rax
  _QWORD *v28; // [rsp+0h] [rbp-A0h]
  unsigned int v29; // [rsp+8h] [rbp-98h]
  _QWORD *v30; // [rsp+8h] [rbp-98h]
  _QWORD *v31; // [rsp+10h] [rbp-90h]
  unsigned int v32; // [rsp+10h] [rbp-90h]
  _QWORD *v33; // [rsp+18h] [rbp-88h]
  _QWORD *v34; // [rsp+18h] [rbp-88h]
  unsigned int v35; // [rsp+18h] [rbp-88h]
  _QWORD v38[4]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v39; // [rsp+50h] [rbp-50h] BYREF

  if ( *(_QWORD *)(a2 + 32) )
    return;
  v5 = a3 + 128;
  v9 = a3 + 128;
  v12 = sub_16D19C0(a3 + 128, a4, a5);
  v13 = (_QWORD *)(*(_QWORD *)(a3 + 128) + 8LL * v12);
  if ( *v13 )
  {
    if ( *v13 != -8 )
    {
      v14 = sub_16E8CB0(v9, a4, v12);
      v15 = sub_16E7EE0(v14, *(const char **)a1, *(_QWORD *)(a1 + 8));
      v16 = sub_1263B40(v15, ": CommandLine Error: Option '");
      v17 = sub_1549FF0(v16, a4, a5);
      sub_1263B40(v17, "' registered more than once!\n");
      sub_16BD130("inconsistency in registered CommandLine options", 1);
    }
    --*(_DWORD *)(a3 + 144);
  }
  v28 = v13;
  v29 = v12;
  v18 = malloc(a5 + 17);
  v19 = v29;
  v20 = v28;
  v21 = (_QWORD *)v18;
  if ( !v18 )
  {
    if ( a5 == -17 )
    {
      v26 = malloc(1u);
      v19 = v29;
      v20 = v28;
      if ( v26 )
      {
        v22 = (void *)(v26 + 16);
        v21 = (_QWORD *)v26;
        goto LABEL_19;
      }
    }
    v30 = v20;
    v32 = v19;
    sub_16BD1C0("Allocation failed");
    v19 = v32;
    v20 = v30;
  }
  v22 = v21 + 2;
  if ( a5 + 1 > 1 )
  {
LABEL_19:
    v31 = v20;
    v35 = v19;
    v27 = memcpy(v22, a4, a5);
    v20 = v31;
    v19 = v35;
    v22 = v27;
  }
  *((_BYTE *)v22 + a5) = 0;
  *v21 = a5;
  v21[1] = a2;
  *v20 = v21;
  ++*(_DWORD *)(a3 + 140);
  sub_16D1CD0(v5, v19);
  if ( a3 == sub_16B4B80((__int64)&unk_4FA0170) )
  {
    sub_16B5AB0(v38, (__int64 *)(a1 + 240), *(_QWORD **)(a1 + 256));
    v23 = (_QWORD *)v38[0];
    v33 = (_QWORD *)v38[1];
    sub_16B55A0(&v39, (__int64 *)(a1 + 240));
    v24 = v39;
    v25 = v33;
    while ( v23 != v24 )
    {
      if ( a3 != *v23 )
      {
        v34 = v25;
        sub_16B7D30(a1, a2, *v23, a4, a5);
        v25 = v34;
      }
      do
        ++v23;
      while ( v23 != v25 && *v23 >= 0xFFFFFFFFFFFFFFFELL );
    }
  }
}
