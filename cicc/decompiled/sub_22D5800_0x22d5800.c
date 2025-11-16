// Function: sub_22D5800
// Address: 0x22d5800
//
__int64 __fastcall sub_22D5800(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  _DWORD *v13; // rdx
  void *v14; // rdx
  int v15; // edx
  __int64 v16; // rsi
  int v17; // edx
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned int v23; // esi
  int *v24; // rcx
  int v25; // r8d
  __int64 v26; // rax
  __int64 *v27; // rcx
  __int64 *v28; // r13
  __int64 *v29; // r14
  _BYTE *v30; // rax
  _BYTE *v31; // r8
  _WORD *v32; // rdx
  __int64 v33; // r12
  _BYTE *v34; // rax
  __int64 v35; // rax
  int v36; // ecx
  void *v37; // rdx
  int v38; // eax
  int v39; // r10d
  int v40; // r8d
  __int64 v41; // [rsp+8h] [rbp-58h]
  __int64 v42; // [rsp+10h] [rbp-50h]
  _BYTE *v43; // [rsp+18h] [rbp-48h]
  __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 136);
  v3 = *(_QWORD *)(v2 + 80);
  result = v2 + 72;
  v41 = result;
  v42 = v3;
  if ( v3 != result )
  {
    v5 = a1;
    while ( 1 )
    {
      v7 = v42 - 24;
      if ( !v42 )
        v7 = 0;
      v8 = sub_AA5930(v7);
      v45 = v9;
      v10 = v8;
      if ( v8 != v9 )
        break;
LABEL_45:
      result = *(_QWORD *)(v42 + 8);
      v42 = result;
      if ( v41 == result )
        return result;
    }
    while ( 1 )
    {
      v13 = *(_DWORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v13 <= 3u )
      {
        sub_CB6200(a2, (unsigned __int8 *)"PHI ", 4u);
      }
      else
      {
        *v13 = 541673552;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      sub_A5BF40((unsigned __int8 *)v10, a2, 0, 0);
      v14 = *(void **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v14 <= 0xCu )
      {
        sub_CB6200(a2, " has values:\n", 0xDu);
      }
      else
      {
        qmemcpy(v14, " has values:\n", 13);
        *(_QWORD *)(a2 + 32) += 13LL;
      }
      v15 = *(_DWORD *)(v5 + 32);
      v16 = *(_QWORD *)(v5 + 16);
      if ( v15 )
      {
        v17 = v15 - 1;
        v18 = v17 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v19 = (__int64 *)(v16 + 16LL * v18);
        v20 = *v19;
        if ( v10 == *v19 )
        {
LABEL_20:
          v15 = *((_DWORD *)v19 + 2);
        }
        else
        {
          v38 = 1;
          while ( v20 != -4096 )
          {
            v40 = v38 + 1;
            v18 = v17 & (v38 + v18);
            v19 = (__int64 *)(v16 + 16LL * v18);
            v20 = *v19;
            if ( *v19 == v10 )
              goto LABEL_20;
            v38 = v40;
          }
          v15 = 0;
        }
      }
      v21 = *(unsigned int *)(v5 + 64);
      v22 = *(_QWORD *)(v5 + 48);
      if ( !(_DWORD)v21 )
        goto LABEL_42;
      v23 = (v21 - 1) & (37 * v15);
      v24 = (int *)(v22 + 88LL * v23);
      v25 = *v24;
      if ( v15 != *v24 )
        break;
LABEL_23:
      if ( v24 == (int *)(v22 + 88 * v21) )
        goto LABEL_42;
      v26 = (unsigned int)v24[12];
      if ( (_DWORD)v26 )
      {
        v27 = (__int64 *)*((_QWORD *)v24 + 5);
        v44 = v5;
        v28 = &v27[v26];
        v29 = v27;
        while ( 1 )
        {
          v31 = (_BYTE *)*v29;
          if ( *(_BYTE *)*v29 > 0x1Cu )
            break;
          v32 = *(_WORD **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v32 <= 1u )
          {
            v43 = (_BYTE *)*v29;
            v35 = sub_CB6200(a2, (unsigned __int8 *)"  ", 2u);
            v31 = v43;
            v33 = v35;
          }
          else
          {
            v33 = a2;
            *v32 = 8224;
            *(_QWORD *)(a2 + 32) += 2LL;
          }
          sub_A69870((__int64)v31, (_BYTE *)v33, 0);
          v34 = *(_BYTE **)(v33 + 32);
          if ( *(_BYTE **)(v33 + 24) == v34 )
          {
            sub_CB6200(v33, (unsigned __int8 *)"\n", 1u);
LABEL_28:
            if ( v28 == ++v29 )
              goto LABEL_34;
          }
          else
          {
            ++v29;
            *v34 = 10;
            ++*(_QWORD *)(v33 + 32);
            if ( v28 == v29 )
            {
LABEL_34:
              v5 = v44;
              goto LABEL_9;
            }
          }
        }
        sub_A69870(*v29, (_BYTE *)a2, 0);
        v30 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v30 )
        {
          sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v30 = 10;
          ++*(_QWORD *)(a2 + 32);
        }
        goto LABEL_28;
      }
      v11 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v11) <= 6 )
      {
        sub_CB6200(a2, "  NONE\n", 7u);
      }
      else
      {
        *(_DWORD *)v11 = 1330520096;
        *(_WORD *)(v11 + 4) = 17742;
        *(_BYTE *)(v11 + 6) = 10;
        *(_QWORD *)(a2 + 32) += 7LL;
      }
LABEL_9:
      if ( !v10 )
        BUG();
      v12 = *(_QWORD *)(v10 + 32);
      if ( !v12 )
        BUG();
      v10 = 0;
      if ( *(_BYTE *)(v12 - 24) == 84 )
        v10 = v12 - 24;
      if ( v45 == v10 )
        goto LABEL_45;
    }
    v36 = 1;
    while ( v25 != -1 )
    {
      v39 = v36 + 1;
      v23 = (v21 - 1) & (v36 + v23);
      v24 = (int *)(v22 + 88LL * v23);
      v25 = *v24;
      if ( v15 == *v24 )
        goto LABEL_23;
      v36 = v39;
    }
LABEL_42:
    v37 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v37 <= 9u )
    {
      sub_CB6200(a2, "  UNKNOWN\n", 0xAu);
    }
    else
    {
      qmemcpy(v37, "  UNKNOWN\n", 10);
      *(_QWORD *)(a2 + 32) += 10LL;
    }
    goto LABEL_9;
  }
  return result;
}
