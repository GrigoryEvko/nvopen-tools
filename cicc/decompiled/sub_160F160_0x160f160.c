// Function: sub_160F160
// Address: 0x160f160
//
void __fastcall sub_160F160(__int64 a1, __int64 a2, __int64 a3, int a4, const char *a5, size_t a6)
{
  int v8; // r12d
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  _WORD *v15; // rdx
  __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // rcx
  __int64 v19; // r8
  const char *v20; // rsi
  __int64 v21; // rdx
  const char *v22; // rdi
  __int64 v23; // rax
  const char *v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  const char *v34; // rsi
  __int64 v35; // rdi
  size_t v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // [rsp+0h] [rbp-60h]
  const char *v41[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v42[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( dword_4F9EB40 <= 2 )
    return;
  v8 = a3;
  v11 = sub_16BA580(a1, a2, a3);
  v12 = *(_BYTE **)(v11 + 24);
  if ( *(_BYTE **)(v11 + 16) == v12 )
  {
    v11 = sub_16E7EE0(v11, "[", 1);
  }
  else
  {
    *v12 = 91;
    ++*(_QWORD *)(v11 + 24);
  }
  v13 = sub_220F850(v11);
  v14 = sub_16AF8C0(v11, v13);
  v15 = *(_WORD **)(v14 + 24);
  v16 = v14;
  if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 1u )
  {
    v16 = sub_16E7EE0(v14, "] ", 2);
  }
  else
  {
    *v15 = 8285;
    *(_QWORD *)(v14 + 24) += 2LL;
  }
  v39 = sub_16E7B40(v16, a1);
  v17 = *(_DWORD *)(a1 + 400);
  v41[0] = (const char *)v42;
  sub_2240A50(v41, (unsigned int)(2 * v17 + 1), 32, v18, v19);
  v20 = v41[0];
  sub_16E7EE0(v39, v41[0], v41[1]);
  v22 = v41[0];
  if ( (_QWORD *)v41[0] != v42 )
  {
    v20 = (const char *)(v42[0] + 1LL);
    j_j___libc_free_0(v41[0], v42[0] + 1LL);
  }
  if ( v8 == 1 )
  {
    v38 = sub_16BA580(v22, v20, v21);
    v34 = "Made Modification '";
    v35 = v38;
  }
  else if ( v8 == 2 )
  {
    v37 = sub_16BA580(v22, v20, v21);
    v34 = " Freeing Pass '";
    v35 = v37;
  }
  else
  {
    if ( v8 )
      goto LABEL_11;
    v33 = sub_16BA580(v22, v20, v21);
    v34 = "Executing Pass '";
    v35 = v33;
  }
  v22 = (const char *)sub_1263B40(v35, v34);
  v20 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
  sub_1549FF0((__int64)v22, v20, v36);
LABEL_11:
  switch ( a4 )
  {
    case 3:
      v31 = sub_16BA580(v22, v20, v21);
      v24 = "' on BasicBlock '";
      v25 = v31;
      goto LABEL_13;
    case 4:
      v30 = sub_16BA580(v22, v20, v21);
      v24 = "' on Function '";
      v25 = v30;
      goto LABEL_13;
    case 5:
      v29 = sub_16BA580(v22, v20, v21);
      v24 = "' on Module '";
      v25 = v29;
      goto LABEL_13;
    case 6:
      v28 = sub_16BA580(v22, v20, v21);
      v24 = "' on Region '";
      v25 = v28;
      goto LABEL_13;
    case 7:
      v23 = sub_16BA580(v22, v20, v21);
      v24 = "' on Loop '";
      v25 = v23;
      goto LABEL_13;
    case 8:
      v32 = sub_16BA580(v22, v20, v21);
      v24 = "' on Call Graph Nodes '";
      v25 = v32;
LABEL_13:
      v26 = sub_1263B40(v25, v24);
      v27 = sub_1549FF0(v26, a5, a6);
      sub_1263B40(v27, "'...\n");
      break;
    default:
      return;
  }
}
