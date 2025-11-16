// Function: sub_2D27C30
// Address: 0x2d27c30
//
__int64 __fastcall sub_2D27C30(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r8
  __int64 v5; // r15
  __int64 *v6; // r13
  unsigned int i; // r12d
  _BYTE *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  _WORD *v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int8 v14; // dl
  __int64 v15; // rax
  __int64 v16; // rdi
  void *v17; // rax
  size_t v18; // rdx
  void *v19; // rdi
  const char *v20; // r14
  void *v21; // rdx
  __int64 v22; // rsi
  _BYTE *v23; // rax
  unsigned int *v24; // r12
  unsigned int *v25; // r13
  unsigned int *v26; // rsi
  __int64 result; // rax
  __int64 v28; // r13
  _BYTE *v29; // rax
  __int64 v30; // r12
  const char *v31; // rax
  size_t v32; // rdx
  _WORD *v33; // rdi
  unsigned __int8 *v34; // rsi
  unsigned __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // r13
  int v38; // ecx
  __int64 v39; // rsi
  int v40; // ecx
  __int64 v41; // r8
  __int64 *v42; // rdi
  __int64 v43; // r9
  __int64 *v44; // rax
  unsigned int *v45; // r12
  unsigned int *v46; // r15
  unsigned int *v47; // rsi
  _BYTE *v48; // rax
  unsigned __int64 v49; // r14
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  int v54; // edx
  __int64 v55; // r11
  unsigned int v56; // edx
  int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rax
  int v60; // r10d
  __int64 v61; // [rsp+0h] [rbp-50h]
  __int64 v63; // [rsp+8h] [rbp-48h]
  __int64 j; // [rsp+10h] [rbp-40h]
  __int64 v65; // [rsp+10h] [rbp-40h]
  size_t v66; // [rsp+10h] [rbp-40h]
  size_t v67; // [rsp+10h] [rbp-40h]

  sub_904010(a2, "=== Variables ===\n");
  v4 = *a1;
  v5 = *a1 + 40LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v5 )
  {
    v6 = (__int64 *)(v4 + 40);
    for ( i = 0; (__int64 *)v5 != v6; v6 += 5 )
    {
      if ( ++i )
      {
        v8 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v8 )
        {
          v9 = sub_CB6200(a2, (unsigned __int8 *)"[", 1u);
        }
        else
        {
          *v8 = 91;
          v9 = a2;
          ++*(_QWORD *)(a2 + 32);
        }
        v10 = sub_CB59D0(v9, i);
        v11 = *(_WORD **)(v10 + 32);
        v12 = v10;
        if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 1u )
        {
          v12 = sub_CB6200(v10, (unsigned __int8 *)"] ", 2u);
        }
        else
        {
          *v11 = 8285;
          *(_QWORD *)(v10 + 32) += 2LL;
        }
        v13 = *v6;
        v14 = *(_BYTE *)(*v6 - 16);
        if ( (v14 & 2) != 0 )
          v15 = *(_QWORD *)(v13 - 32);
        else
          v15 = v13 - 16 - 8LL * ((v14 >> 2) & 0xF);
        v16 = *(_QWORD *)(v15 + 8);
        if ( v16 )
        {
          v17 = (void *)sub_B91420(v16);
          v19 = *(void **)(v12 + 32);
          if ( v18 <= *(_QWORD *)(v12 + 24) - (_QWORD)v19 )
          {
            if ( v18 )
            {
              v67 = v18;
              memcpy(v19, v17, v18);
              *(_QWORD *)(v12 + 32) += v67;
            }
          }
          else
          {
            sub_CB6200(v12, (unsigned __int8 *)v17, v18);
          }
        }
        if ( *((_BYTE *)v6 + 24) )
        {
          v49 = v6[2];
          v65 = v6[1];
          v50 = sub_904010(a2, " bits [");
          v51 = sub_CB59D0(v50, v49);
          v52 = sub_904010(v51, ", ");
          v53 = sub_CB59D0(v52, v49 + v65);
          sub_904010(v53, ")");
        }
        v20 = (const char *)v6[4];
        if ( v20 )
        {
          v21 = *(void **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v21 <= 0xBu )
          {
            v22 = sub_CB6200(a2, " inlined-at ", 0xCu);
          }
          else
          {
            v22 = a2;
            qmemcpy(v21, " inlined-at ", 12);
            *(_QWORD *)(a2 + 32) += 12LL;
          }
          sub_A61DE0(v20, v22, 0);
        }
        v23 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v23 )
        {
          sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v23 = 10;
          ++*(_QWORD *)(a2 + 32);
        }
      }
    }
  }
  sub_904010(a2, "=== Single location vars ===\n");
  v24 = (unsigned int *)a1[7];
  v25 = &v24[8 * *((unsigned int *)a1 + 26)];
  while ( v24 != v25 )
  {
    v26 = v24;
    v24 += 8;
    sub_2D257B0(a2, v26);
  }
  sub_904010(a2, "=== In-line variable defs ===");
  result = a3 + 72;
  v61 = result;
  v63 = *(_QWORD *)(a3 + 80);
  if ( result != v63 )
  {
    do
    {
      v28 = v63 - 24;
      if ( !v63 )
        v28 = 0;
      v29 = *(_BYTE **)(a2 + 32);
      if ( *(_BYTE **)(a2 + 24) == v29 )
      {
        v30 = sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v29 = 10;
        v30 = a2;
        ++*(_QWORD *)(a2 + 32);
      }
      v31 = sub_BD5D20(v28);
      v33 = *(_WORD **)(v30 + 32);
      v34 = (unsigned __int8 *)v31;
      v35 = *(_QWORD *)(v30 + 24) - (_QWORD)v33;
      if ( v35 < v32 )
      {
        v58 = sub_CB6200(v30, v34, v32);
        v33 = *(_WORD **)(v58 + 32);
        v30 = v58;
        v35 = *(_QWORD *)(v58 + 24) - (_QWORD)v33;
      }
      else if ( v32 )
      {
        v66 = v32;
        memcpy(v33, v34, v32);
        v59 = *(_QWORD *)(v30 + 24);
        v33 = (_WORD *)(v66 + *(_QWORD *)(v30 + 32));
        *(_QWORD *)(v30 + 32) = v33;
        v35 = v59 - (_QWORD)v33;
      }
      if ( v35 <= 1 )
      {
        sub_CB6200(v30, (unsigned __int8 *)":\n", 2u);
      }
      else
      {
        *v33 = 2618;
        *(_QWORD *)(v30 + 32) += 2LL;
      }
      v36 = *(_QWORD *)(v28 + 56);
      for ( j = v28 + 48; j != v36; v36 = *(_QWORD *)(v36 + 8) )
      {
        v37 = v36 - 24;
        if ( !v36 )
          v37 = 0;
        v38 = *((_DWORD *)a1 + 34);
        v39 = a1[15];
        if ( v38 )
        {
          v40 = v38 - 1;
          v41 = v40 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
          v42 = (__int64 *)(v39 + 16 * v41);
          v43 = *v42;
          v44 = v42;
          if ( v37 == *v42 )
          {
LABEL_39:
            v45 = (unsigned int *)a1[7];
            v46 = &v45[8 * *((unsigned int *)v44 + 2)];
          }
          else
          {
            v55 = *v42;
            v56 = v40 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
            v57 = 1;
            while ( v55 != -4096 )
            {
              v60 = v57 + 1;
              v56 = v40 & (v57 + v56);
              v44 = (__int64 *)(v39 + 16LL * v56);
              v55 = *v44;
              if ( v37 == *v44 )
                goto LABEL_39;
              v57 = v60;
            }
            v45 = (unsigned int *)a1[7];
            v46 = v45;
            LODWORD(v41) = v40 & (((unsigned int)v37 >> 4) ^ ((unsigned int)v37 >> 9));
            v42 = (__int64 *)(v39 + 16LL * (unsigned int)v41);
            v43 = *v42;
          }
          if ( v43 == v37 )
          {
LABEL_41:
            v45 += 8 * *((unsigned int *)v42 + 3);
          }
          else
          {
            v54 = 1;
            while ( v43 != -4096 )
            {
              LODWORD(v41) = v40 & (v54 + v41);
              v42 = (__int64 *)(v39 + 16LL * (unsigned int)v41);
              v43 = *v42;
              if ( v37 == *v42 )
                goto LABEL_41;
              ++v54;
            }
          }
          while ( v46 != v45 )
          {
            v47 = v46;
            v46 += 8;
            sub_2D257B0(a2, v47);
          }
        }
        sub_A69870(v37, (_BYTE *)a2, 0);
        v48 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v48 )
        {
          sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v48 = 10;
          ++*(_QWORD *)(a2 + 32);
        }
      }
      result = *(_QWORD *)(v63 + 8);
      v63 = result;
    }
    while ( v61 != result );
  }
  return result;
}
