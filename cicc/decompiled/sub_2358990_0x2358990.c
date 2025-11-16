// Function: sub_2358990
// Address: 0x2358990
//
void __fastcall sub_2358990(unsigned __int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // ecx
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  int v14; // esi
  unsigned __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // esi
  unsigned __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // esi
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // rdi
  __int64 v27; // [rsp+8h] [rbp-C8h] BYREF
  int v28; // [rsp+18h] [rbp-B8h] BYREF
  unsigned __int64 v29; // [rsp+20h] [rbp-B0h]
  int *v30; // [rsp+28h] [rbp-A8h]
  int *v31; // [rsp+30h] [rbp-A0h]
  __int64 v32; // [rsp+38h] [rbp-98h]
  int v33; // [rsp+48h] [rbp-88h] BYREF
  unsigned __int64 v34; // [rsp+50h] [rbp-80h]
  int *v35; // [rsp+58h] [rbp-78h]
  int *v36; // [rsp+60h] [rbp-70h]
  __int64 v37; // [rsp+68h] [rbp-68h]
  int v38; // [rsp+78h] [rbp-58h] BYREF
  unsigned __int64 v39; // [rsp+80h] [rbp-50h]
  int *v40; // [rsp+88h] [rbp-48h]
  int *v41; // [rsp+90h] [rbp-40h]
  __int64 v42; // [rsp+98h] [rbp-38h]
  char v43; // [rsp+A0h] [rbp-30h]

  v2 = *(_QWORD *)(a2 + 16);
  if ( v2 )
  {
    v3 = *(_DWORD *)(a2 + 8);
    v29 = *(_QWORD *)(a2 + 16);
    v28 = v3;
    v30 = *(int **)(a2 + 24);
    v31 = *(int **)(a2 + 32);
    *(_QWORD *)(v2 + 8) = &v28;
    v4 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(a2 + 16) = 0;
    v32 = v4;
    v5 = *(_QWORD *)(a2 + 64);
    *(_QWORD *)(a2 + 24) = a2 + 8;
    *(_QWORD *)(a2 + 32) = a2 + 8;
    *(_QWORD *)(a2 + 40) = 0;
    if ( v5 )
      goto LABEL_3;
LABEL_23:
    v8 = *(_QWORD *)(a2 + 112);
    v33 = 0;
    v34 = 0;
    v35 = &v33;
    v36 = &v33;
    v37 = 0;
    if ( v8 )
      goto LABEL_4;
    goto LABEL_24;
  }
  v5 = *(_QWORD *)(a2 + 64);
  v28 = 0;
  v29 = 0;
  v30 = &v28;
  v31 = &v28;
  v32 = 0;
  if ( !v5 )
    goto LABEL_23;
LABEL_3:
  v6 = *(_DWORD *)(a2 + 56);
  v34 = v5;
  v33 = v6;
  v35 = *(int **)(a2 + 72);
  v36 = *(int **)(a2 + 80);
  *(_QWORD *)(v5 + 8) = &v33;
  v7 = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a2 + 64) = 0;
  v37 = v7;
  v8 = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a2 + 72) = a2 + 56;
  *(_QWORD *)(a2 + 80) = a2 + 56;
  *(_QWORD *)(a2 + 88) = 0;
  if ( v8 )
  {
LABEL_4:
    v9 = *(_DWORD *)(a2 + 104);
    v39 = v8;
    v38 = v9;
    v40 = *(int **)(a2 + 120);
    v41 = *(int **)(a2 + 128);
    *(_QWORD *)(v8 + 8) = &v38;
    v10 = *(_QWORD *)(a2 + 136);
    *(_QWORD *)(a2 + 112) = 0;
    v42 = v10;
    *(_QWORD *)(a2 + 120) = a2 + 104;
    *(_QWORD *)(a2 + 128) = a2 + 104;
    *(_QWORD *)(a2 + 136) = 0;
    goto LABEL_5;
  }
LABEL_24:
  v38 = 0;
  v39 = 0;
  v40 = &v38;
  v41 = &v38;
  v42 = 0;
LABEL_5:
  v43 = *(_BYTE *)(a2 + 144);
  v11 = sub_22077B0(0xA0u);
  if ( v11 )
  {
    v12 = v11 + 16;
    *(_QWORD *)v11 = &unk_4A0D138;
    v13 = v29;
    if ( v29 )
    {
      v14 = v28;
      *(_QWORD *)(v11 + 24) = v29;
      *(_DWORD *)(v11 + 16) = v14;
      *(_QWORD *)(v11 + 32) = v30;
      *(_QWORD *)(v11 + 40) = v31;
      *(_QWORD *)(v13 + 8) = v12;
      v29 = 0;
      *(_QWORD *)(v11 + 48) = v32;
      v30 = &v28;
      v31 = &v28;
      v32 = 0;
    }
    else
    {
      *(_DWORD *)(v11 + 16) = 0;
      *(_QWORD *)(v11 + 24) = 0;
      *(_QWORD *)(v11 + 32) = v12;
      *(_QWORD *)(v11 + 40) = v12;
      *(_QWORD *)(v11 + 48) = 0;
    }
    v15 = v34;
    v16 = v11 + 64;
    if ( v34 )
    {
      v17 = v33;
      *(_QWORD *)(v11 + 72) = v34;
      *(_DWORD *)(v11 + 64) = v17;
      *(_QWORD *)(v11 + 80) = v35;
      *(_QWORD *)(v11 + 88) = v36;
      *(_QWORD *)(v15 + 8) = v16;
      v34 = 0;
      *(_QWORD *)(v11 + 96) = v37;
      v35 = &v33;
      v36 = &v33;
      v37 = 0;
    }
    else
    {
      *(_DWORD *)(v11 + 64) = 0;
      *(_QWORD *)(v11 + 72) = 0;
      *(_QWORD *)(v11 + 80) = v16;
      *(_QWORD *)(v11 + 88) = v16;
      *(_QWORD *)(v11 + 96) = 0;
    }
    v18 = v39;
    v19 = v11 + 112;
    if ( v39 )
    {
      v20 = v38;
      *(_QWORD *)(v11 + 120) = v39;
      *(_DWORD *)(v11 + 112) = v20;
      *(_QWORD *)(v11 + 128) = v40;
      *(_QWORD *)(v11 + 136) = v41;
      *(_QWORD *)(v18 + 8) = v19;
      v39 = 0;
      *(_QWORD *)(v11 + 144) = v42;
      v40 = &v38;
      v41 = &v38;
      v42 = 0;
    }
    else
    {
      *(_DWORD *)(v11 + 112) = 0;
      *(_QWORD *)(v11 + 120) = 0;
      *(_QWORD *)(v11 + 128) = v19;
      *(_QWORD *)(v11 + 136) = v19;
      *(_QWORD *)(v11 + 144) = 0;
    }
    *(_BYTE *)(v11 + 152) = v43;
  }
  v27 = v11;
  sub_2356EF0(a1, (unsigned __int64 *)&v27);
  if ( v27 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
  v21 = v39;
  while ( v21 )
  {
    sub_2307A70(*(_QWORD *)(v21 + 24));
    v22 = v21;
    v21 = *(_QWORD *)(v21 + 16);
    j_j___libc_free_0(v22);
  }
  v23 = v34;
  while ( v23 )
  {
    sub_23076D0(*(_QWORD *)(v23 + 24));
    v24 = v23;
    v23 = *(_QWORD *)(v23 + 16);
    j_j___libc_free_0(v24);
  }
  v25 = v29;
  while ( v25 )
  {
    sub_23078A0(*(_QWORD *)(v25 + 24));
    v26 = v25;
    v25 = *(_QWORD *)(v25 + 16);
    j_j___libc_free_0(v26);
  }
}
