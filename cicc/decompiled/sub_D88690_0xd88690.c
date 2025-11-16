// Function: sub_D88690
// Address: 0xd88690
//
__int64 __fastcall sub_D88690(_QWORD *a1, __int64 a2, unsigned __int8 *a3, size_t a4, __int64 a5)
{
  __int64 v9; // rax
  void *v10; // rdi
  __int64 v11; // r12
  const char *v12; // rsi
  __int64 v13; // r12
  const char *v14; // rsi
  __int64 v15; // rax
  char *v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // r12
  _QWORD *v19; // r14
  unsigned int v20; // r15d
  const char *v21; // rax
  size_t v22; // rdx
  _DWORD *v23; // rdi
  unsigned __int8 *v24; // rsi
  unsigned __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rdi
  _BYTE *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r15
  __int64 v31; // rbx
  __int64 i; // r12
  __int64 v33; // rax
  _DWORD *v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned __int64 v37; // r8
  _QWORD *v38; // r14
  _QWORD *v39; // rax
  _QWORD *v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // r14
  unsigned __int8 *v44; // rax
  size_t v45; // rdx
  void *v46; // rdi
  __int64 v47; // r8
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 result; // rax
  __int64 v54; // [rsp+10h] [rbp-90h]
  size_t v55; // [rsp+10h] [rbp-90h]
  __int64 v56; // [rsp+18h] [rbp-88h]
  size_t v58; // [rsp+28h] [rbp-78h]
  __int64 v59[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v60[2]; // [rsp+40h] [rbp-60h] BYREF
  char v61; // [rsp+50h] [rbp-50h]
  _QWORD v62[2]; // [rsp+58h] [rbp-48h] BYREF
  _QWORD *v63; // [rsp+68h] [rbp-38h] BYREF

  v9 = sub_904010(a2, "  @");
  v10 = *(void **)(v9 + 32);
  v11 = v9;
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 < a4 )
  {
    v11 = sub_CB6200(v9, a3, a4);
LABEL_3:
    if ( a5 )
      goto LABEL_4;
LABEL_33:
    v35 = sub_904010(v11, " dso_preemptable");
    v36 = sub_904010(v35, byte_3F871B3);
    sub_904010(v36, "\n");
    v16 = "    args uses:\n";
    sub_904010(a2, "    args uses:\n");
    v18 = a1[9];
    v19 = a1 + 7;
    if ( (_QWORD *)v18 != a1 + 7 )
      goto LABEL_20;
    return sub_904010(a2, "    allocas uses:\n");
  }
  if ( !a4 )
    goto LABEL_3;
  memcpy(v10, a3, a4);
  *(_QWORD *)(v11 + 32) += a4;
  if ( !a5 )
    goto LABEL_33;
LABEL_4:
  v12 = " dso_preemptable";
  if ( (*(_BYTE *)(a5 + 33) & 0x40) != 0 )
    v12 = byte_3F871B3;
  v13 = sub_904010(v11, v12);
  v14 = " interposable";
  if ( !(unsigned __int8)sub_B2F6B0(a5) )
    v14 = byte_3F871B3;
  v15 = sub_904010(v13, v14);
  sub_904010(v15, "\n");
  v16 = "    args uses:\n";
  sub_904010(a2, "    args uses:\n");
  v18 = a1[9];
  v19 = a1 + 7;
  if ( a1 + 7 == (_QWORD *)v18 )
  {
    result = sub_904010(a2, "    allocas uses:\n");
    goto LABEL_26;
  }
  do
  {
    while ( 1 )
    {
LABEL_20:
      v29 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v29) > 5 )
      {
        *(_DWORD *)v29 = 538976288;
        *(_WORD *)(v29 + 4) = 8224;
        *(_QWORD *)(a2 + 32) += 6LL;
        if ( !a5 )
          goto LABEL_22;
      }
      else
      {
        v16 = "      ";
        sub_CB6200(a2, (unsigned __int8 *)"      ", 6u);
        if ( !a5 )
        {
LABEL_22:
          v59[1] = 6;
          v59[0] = (__int64)"arg{0}";
          v60[0] = (__int64)&v63;
          v60[1] = 1;
          v61 = 1;
          v62[0] = &unk_49DC910;
          v62[1] = v18 + 32;
          v63 = v62;
          sub_CB6840(a2, (__int64)v59);
          v23 = *(_DWORD **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v23 > 3u )
            goto LABEL_17;
          goto LABEL_23;
        }
      }
      v20 = *(_DWORD *)(v18 + 32);
      if ( (*(_BYTE *)(a5 + 2) & 1) != 0 )
        sub_B2C6D0(a5, (__int64)v16, v29, v17);
      v21 = sub_BD5D20(*(_QWORD *)(a5 + 96) + 40LL * v20);
      v23 = *(_DWORD **)(a2 + 32);
      v24 = (unsigned __int8 *)v21;
      v25 = *(_QWORD *)(a2 + 24) - (_QWORD)v23;
      if ( v25 < v22 )
      {
        sub_CB6200(a2, v24, v22);
        v23 = *(_DWORD **)(a2 + 32);
        v25 = *(_QWORD *)(a2 + 24) - (_QWORD)v23;
      }
      else if ( v22 )
      {
        v58 = v22;
        memcpy(v23, v24, v22);
        v33 = *(_QWORD *)(a2 + 24);
        v34 = (_DWORD *)(*(_QWORD *)(a2 + 32) + v58);
        *(_QWORD *)(a2 + 32) = v34;
        v23 = v34;
        v25 = v33 - (_QWORD)v34;
      }
      if ( v25 > 3 )
      {
LABEL_17:
        *v23 = 540695899;
        v26 = a2;
        *(_QWORD *)(a2 + 32) += 4LL;
        goto LABEL_18;
      }
LABEL_23:
      v26 = sub_CB6200(a2, "[]: ", 4u);
LABEL_18:
      v16 = (char *)(v18 + 40);
      v27 = sub_D86260(v26, v18 + 40);
      v28 = *(_BYTE **)(v27 + 32);
      if ( *(_BYTE **)(v27 + 24) == v28 )
        break;
      *v28 = 10;
      ++*(_QWORD *)(v27 + 32);
      v18 = sub_220EF30(v18);
      if ( (_QWORD *)v18 == v19 )
        goto LABEL_25;
    }
    v16 = "\n";
    sub_CB6200(v27, (unsigned __int8 *)"\n", 1u);
    v18 = sub_220EF30(v18);
  }
  while ( (_QWORD *)v18 != v19 );
LABEL_25:
  result = sub_904010(a2, "    allocas uses:\n");
  if ( a5 )
  {
LABEL_26:
    v30 = a5 + 72;
    v31 = *(_QWORD *)(a5 + 80);
    if ( v30 == v31 )
    {
      i = 0;
    }
    else
    {
      while ( 1 )
      {
        if ( !v31 )
LABEL_63:
          BUG();
        i = *(_QWORD *)(v31 + 32);
        result = v31 + 24;
        if ( i != v31 + 24 )
          break;
        v31 = *(_QWORD *)(v31 + 8);
        if ( v30 == v31 )
          return result;
      }
    }
    if ( v30 != v31 )
    {
      if ( !i )
LABEL_60:
        BUG();
      while ( 1 )
      {
        if ( *(_BYTE *)(i - 24) == 60 )
        {
          v37 = i - 24;
          v38 = a1 + 1;
          v39 = (_QWORD *)a1[2];
          if ( v39 )
          {
            v40 = a1 + 1;
            do
            {
              while ( 1 )
              {
                v41 = v39[2];
                v42 = v39[3];
                if ( v39[4] >= v37 )
                  break;
                v39 = (_QWORD *)v39[3];
                if ( !v42 )
                  goto LABEL_44;
              }
              v40 = v39;
              v39 = (_QWORD *)v39[2];
            }
            while ( v41 );
LABEL_44:
            if ( v38 != v40 && v40[4] <= v37 )
              v38 = v40;
          }
          v56 = (__int64)(v38 + 5);
          v43 = sub_904010(a2, "      ");
          v44 = (unsigned __int8 *)sub_BD5D20(i - 24);
          v46 = *(void **)(v43 + 32);
          v47 = i - 24;
          if ( *(_QWORD *)(v43 + 24) - (_QWORD)v46 < v45 )
          {
            v52 = sub_CB6200(v43, v44, v45);
            v47 = i - 24;
            v43 = v52;
          }
          else if ( v45 )
          {
            v55 = v45;
            memcpy(v46, v44, v45);
            *(_QWORD *)(v43 + 32) += v55;
            v47 = i - 24;
          }
          v54 = v47;
          v48 = sub_904010(v43, "[");
          sub_D882F0(v59, v54);
          sub_C49420((__int64)v60, v48, 1);
          v49 = sub_904010(v48, "]: ");
          v50 = sub_D86260(v49, v56);
          sub_904010(v50, "\n");
          sub_969240(v60);
          sub_969240(v59);
        }
        for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v31 + 32) )
        {
          v51 = v31 - 24;
          if ( !v31 )
            v51 = 0;
          result = v51 + 48;
          if ( i != result )
            break;
          v31 = *(_QWORD *)(v31 + 8);
          if ( v30 == v31 )
            return result;
          if ( !v31 )
            goto LABEL_63;
        }
        if ( v30 == v31 )
          break;
        if ( !i )
          goto LABEL_60;
      }
    }
  }
  return result;
}
