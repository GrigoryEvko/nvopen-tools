// Function: sub_23B44D0
// Address: 0x23b44d0
//
bool __fastcall sub_23B44D0(__int64 *a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // rax
  void *(*v4)(); // rdx
  __int64 v5; // rbx
  _QWORD *v6; // r12
  _QWORD *v7; // rbx
  __int64 v8; // rdi
  char *v9; // rax
  __int64 v10; // rdx
  bool result; // al
  void *v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // rax
  __int64 (*v15)(void); // rdx
  __int64 v16; // r14
  char *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  char *v20; // rax
  __int64 v21; // rdx
  void *v22; // rax
  void *(*v23)(); // rax
  void *v24; // rax
  __int64 v25; // r14
  __int64 v26; // r13
  __int64 v27; // r13
  _QWORD *v28; // r13
  char *v29; // rax
  __int64 v30; // rdx
  char *v31; // rax
  __int64 v32; // rdx
  char *v33; // rax
  __int64 v34; // rdx
  char *v35; // rax
  __int64 v36; // rdx
  void *(*v37)(); // rax
  void *v38; // rax
  __int64 v39; // rbx
  void *(*v40)(); // rax
  char *v41; // rax
  __int64 v42; // rdx
  char *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r12
  __int64 v46[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_23B2720(v46, a1);
  v2 = (_QWORD *)v46[0];
  if ( !v46[0] )
  {
LABEL_17:
    sub_23B2720(v46, a1);
    v13 = (_QWORD *)v46[0];
    if ( !v46[0] )
      goto LABEL_33;
    v14 = *(_QWORD *)v46[0];
    v15 = *(__int64 (**)(void))(*(_QWORD *)v46[0] + 24LL);
    if ( (char *)v15 == (char *)sub_23AE340 )
    {
      if ( &unk_4CDFBF8 == &unk_4C5D161 )
        goto LABEL_20;
    }
    else
    {
      v22 = (void *)v15();
      v13 = (_QWORD *)v46[0];
      if ( v22 == &unk_4C5D161 )
      {
LABEL_20:
        v16 = v13[1];
        (*(void (__fastcall **)(_QWORD *))(*v13 + 8LL))(v13);
        if ( v16 )
        {
          v17 = (char *)sub_BD5D20(v16);
          return sub_BC63A0(v17, v18);
        }
LABEL_33:
        sub_23B2720(v46, a1);
        if ( v46[0]
          && ((v23 = *(void *(**)())(*(_QWORD *)v46[0] + 24LL), v23 != sub_23AE340)
            ? (v24 = v23())
            : (v24 = &unk_4CDFBF8),
              v24 == &unk_4C5D118) )
        {
          v25 = *(_QWORD *)(v46[0] + 8);
          sub_23B42E0(v46);
          if ( v25 )
          {
            v7 = *(_QWORD **)(v25 + 8);
            v26 = 8LL * *(unsigned int *)(v25 + 16);
            v6 = &v7[(unsigned __int64)v26 / 8];
            v19 = v26 >> 3;
            v27 = v26 >> 5;
            if ( v27 )
            {
              v28 = &v7[4 * v27];
              while ( 1 )
              {
                v35 = (char *)sub_BD5D20(*(_QWORD *)(*v7 + 8LL));
                if ( sub_BC63A0(v35, v36) )
                  goto LABEL_45;
                v29 = (char *)sub_BD5D20(*(_QWORD *)(v7[1] + 8LL));
                if ( sub_BC63A0(v29, v30) )
                {
                  ++v7;
                  goto LABEL_45;
                }
                v31 = (char *)sub_BD5D20(*(_QWORD *)(v7[2] + 8LL));
                if ( sub_BC63A0(v31, v32) )
                {
                  v7 += 2;
                  goto LABEL_45;
                }
                v33 = (char *)sub_BD5D20(*(_QWORD *)(v7[3] + 8LL));
                if ( sub_BC63A0(v33, v34) )
                {
                  v7 += 3;
                  goto LABEL_45;
                }
                v7 += 4;
                if ( v28 == v7 )
                {
                  v19 = v6 - v7;
                  break;
                }
              }
            }
            if ( v19 != 2 )
            {
              if ( v19 != 3 )
              {
                if ( v19 != 1 )
                  return sub_BC63A0("*", 1);
                goto LABEL_27;
              }
              v41 = (char *)sub_BD5D20(*(_QWORD *)(*v7 + 8LL));
              if ( sub_BC63A0(v41, v42) )
              {
LABEL_45:
                result = 1;
                goto LABEL_11;
              }
              ++v7;
            }
            v43 = (char *)sub_BD5D20(*(_QWORD *)(*v7 + 8LL));
            if ( !sub_BC63A0(v43, v44) )
            {
              ++v7;
LABEL_27:
              v20 = (char *)sub_BD5D20(*(_QWORD *)(*v7 + 8LL));
              if ( !sub_BC63A0(v20, v21) )
                return sub_BC63A0("*", 1);
              goto LABEL_45;
            }
            goto LABEL_45;
          }
        }
        else
        {
          sub_23B42E0(v46);
        }
        sub_23B2720(v46, a1);
        if ( v46[0]
          && ((v37 = *(void *(**)())(*(_QWORD *)v46[0] + 24LL), v37 != sub_23AE340)
            ? (v38 = v37())
            : (v38 = &unk_4CDFBF8),
              v38 == &unk_4C5D160) )
        {
          v39 = *(_QWORD *)(v46[0] + 8);
          sub_23B42E0(v46);
          if ( v39 )
          {
            v17 = (char *)sub_BD5D20(*(_QWORD *)(**(_QWORD **)(v39 + 32) + 72LL));
            return sub_BC63A0(v17, v18);
          }
        }
        else
        {
          sub_23B42E0(v46);
        }
        sub_23B2720(v46, a1);
        if ( v46[0] && (v40 = *(void *(**)())(*(_QWORD *)v46[0] + 24LL), v40 != sub_23AE340) && v40() == &unk_4CDFC40 )
        {
          v45 = *(_QWORD *)(v46[0] + 8);
          sub_23B42E0(v46);
          if ( v45 )
          {
            v17 = (char *)sub_2E791E0(v45);
            return sub_BC63A0(v17, v18);
          }
        }
        else
        {
          sub_23B42E0(v46);
        }
        BUG();
      }
      if ( !v46[0] )
        goto LABEL_33;
      v14 = *(_QWORD *)v46[0];
    }
    (*(void (**)(void))(v14 + 8))();
    goto LABEL_33;
  }
  v3 = *(_QWORD *)v46[0];
  v4 = *(void *(**)())(*(_QWORD *)v46[0] + 24LL);
  if ( v4 == sub_23AE340 )
  {
    if ( &unk_4CDFBF8 == &unk_4C5D162 )
      goto LABEL_4;
LABEL_16:
    (*(void (**)(void))(v3 + 8))();
    goto LABEL_17;
  }
  v12 = v4();
  v2 = (_QWORD *)v46[0];
  if ( v12 != &unk_4C5D162 )
  {
    if ( !v46[0] )
      goto LABEL_17;
    v3 = *(_QWORD *)v46[0];
    goto LABEL_16;
  }
LABEL_4:
  v5 = v2[1];
  (*(void (__fastcall **)(_QWORD *))(*v2 + 8LL))(v2);
  if ( !v5 )
    goto LABEL_17;
  v6 = (_QWORD *)(v5 + 24);
  v7 = *(_QWORD **)(v5 + 32);
  if ( v6 == v7 )
    return sub_BC63A0("*", 1);
  while ( 1 )
  {
    v8 = (__int64)(v7 - 7);
    if ( !v7 )
      v8 = 0;
    v9 = (char *)sub_BD5D20(v8);
    result = sub_BC63A0(v9, v10);
    if ( result )
      break;
    v7 = (_QWORD *)v7[1];
    if ( v6 == v7 )
      return sub_BC63A0("*", 1);
  }
LABEL_11:
  if ( v6 == v7 )
    return sub_BC63A0("*", 1);
  return result;
}
