// Function: sub_B817B0
// Address: 0xb817b0
//
void __fastcall sub_B817B0(__int64 a1, __int64 a2, int a3, int a4, const void *a5, size_t a6)
{
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
  __int64 v21; // rdi
  __int64 v22; // rax
  void *v23; // rdi
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 v26; // rax
  const char *v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // r12
  const void *v30; // rax
  size_t v31; // rdx
  void *v32; // rdi
  __int64 v33; // [rsp+0h] [rbp-60h]
  size_t v34; // [rsp+0h] [rbp-60h]
  _QWORD v36[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v37[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( (int)qword_4F81B88 <= 2 )
    return;
  v11 = sub_C5F790();
  v12 = *(_BYTE **)(v11 + 32);
  if ( *(_BYTE **)(v11 + 24) == v12 )
  {
    v11 = sub_CB6200(v11, "[", 1);
  }
  else
  {
    *v12 = 91;
    ++*(_QWORD *)(v11 + 32);
  }
  v13 = sub_220F850();
  v14 = sub_C4F170(v11, v13);
  v15 = *(_WORD **)(v14 + 32);
  v16 = v14;
  if ( *(_QWORD *)(v14 + 24) - (_QWORD)v15 <= 1u )
  {
    v16 = sub_CB6200(v14, "] ", 2);
  }
  else
  {
    *v15 = 8285;
    *(_QWORD *)(v14 + 32) += 2LL;
  }
  v33 = sub_CB5A80(v16, a1);
  v17 = *(_DWORD *)(a1 + 384);
  v36[0] = v37;
  sub_2240A50(v36, (unsigned int)(2 * v17 + 1), 32, v18, v19);
  sub_CB6200(v33, v36[0], v36[1]);
  if ( (_QWORD *)v36[0] != v37 )
    j_j___libc_free_0(v36[0], v37[0] + 1LL);
  switch ( a3 )
  {
    case 1:
      v27 = "Made Modification '";
      v28 = sub_C5F790();
LABEL_25:
      v29 = sub_904010(v28, v27);
      v30 = (const void *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
      v32 = *(void **)(v29 + 32);
      if ( *(_QWORD *)(v29 + 24) - (_QWORD)v32 < v31 )
      {
        sub_CB6200(v29, v30, v31);
      }
      else if ( v31 )
      {
        v34 = v31;
        memcpy(v32, v30, v31);
        *(_QWORD *)(v29 + 32) += v34;
      }
      break;
    case 2:
      v27 = " Freeing Pass '";
      v28 = sub_C5F790();
      goto LABEL_25;
    case 0:
      v27 = "Executing Pass '";
      v28 = sub_C5F790();
      goto LABEL_25;
  }
  switch ( a4 )
  {
    case 3:
      v20 = "' on Function '";
      v21 = sub_C5F790();
      goto LABEL_13;
    case 4:
      v25 = sub_C5F790();
      v26 = sub_904010(v25, "' on Module '");
      v23 = *(void **)(v26 + 32);
      v24 = v26;
      if ( *(_QWORD *)(v26 + 24) - (_QWORD)v23 < a6 )
        goto LABEL_19;
      goto LABEL_14;
    case 5:
      v20 = "' on Region '";
      v21 = sub_C5F790();
      goto LABEL_13;
    case 6:
      v20 = "' on Loop '";
      v21 = sub_C5F790();
      goto LABEL_13;
    case 7:
      v20 = "' on Call Graph Nodes '";
      v21 = sub_C5F790();
LABEL_13:
      v22 = sub_904010(v21, v20);
      v23 = *(void **)(v22 + 32);
      v24 = v22;
      if ( a6 > *(_QWORD *)(v22 + 24) - (_QWORD)v23 )
      {
LABEL_19:
        v24 = sub_CB6200(v24, a5, a6);
      }
      else
      {
LABEL_14:
        if ( a6 )
        {
          memcpy(v23, a5, a6);
          *(_QWORD *)(v24 + 32) += a6;
        }
      }
      sub_904010(v24, "'...\n");
      break;
    default:
      return;
  }
}
