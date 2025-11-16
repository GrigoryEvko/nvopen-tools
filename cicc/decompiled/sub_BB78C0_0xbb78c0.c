// Function: sub_BB78C0
// Address: 0xbb78c0
//
__int64 __fastcall sub_BB78C0(__int64 a1, _DWORD *a2, size_t a3, const void *a4, size_t a5)
{
  const void *v5; // r15
  bool v8; // zf
  int v9; // ebx
  _DWORD *v10; // rax
  _DWORD *v11; // rdx
  unsigned int v12; // eax
  unsigned int v13; // r14d
  const char *v14; // r10
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  const char *v19; // rdi
  __int64 v20; // rax
  size_t v21; // r9
  const char *v22; // r10
  _QWORD *v23; // rdx
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  _BYTE *v28; // rax
  __int64 v29; // rax
  _WORD *v30; // rdx
  __int64 v31; // rbx
  _DWORD *v32; // rdi
  unsigned __int64 v33; // rax
  _BYTE *v34; // rdi
  _BYTE *v35; // rax
  __int64 v36; // rax
  unsigned int v37; // edx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  const char *v43; // [rsp+8h] [rbp-48h]
  size_t v44; // [rsp+8h] [rbp-48h]
  size_t v45; // [rsp+10h] [rbp-40h]
  const char *v46; // [rsp+10h] [rbp-40h]

  v5 = a2;
  v8 = *(_QWORD *)(a1 + 88) == 0;
  v9 = *(_DWORD *)(a1 + 12) + 1;
  *(_DWORD *)(a1 + 12) = v9;
  if ( v8 )
  {
    v10 = *(_DWORD **)(a1 + 16);
    v11 = &v10[*(unsigned int *)(a1 + 24)];
    if ( v10 != v11 )
    {
      while ( v9 != *v10 )
      {
        if ( v11 == ++v10 )
          goto LABEL_3;
      }
      if ( v11 != v10 )
      {
LABEL_9:
        v12 = (unsigned __int8)qword_4F82028;
LABEL_10:
        v13 = v12;
        if ( !(_BYTE)v12 )
          return v13;
        v13 = 0;
        v14 = "NOT ";
LABEL_24:
        v19 = v14;
        v43 = v14;
        v45 = strlen(v14);
        v20 = sub_CB72A0(v19, a2);
        v21 = v45;
        v22 = v43;
        v23 = *(_QWORD **)(v20 + 32);
        v24 = v20;
        if ( *(_QWORD *)(v20 + 24) - (_QWORD)v23 <= 7u )
        {
          v44 = v45;
          v46 = v22;
          v40 = sub_CB6200(v20, "BISECT: ", 8);
          v22 = v46;
          v21 = v44;
          v24 = v40;
          v25 = *(_BYTE **)(v40 + 32);
        }
        else
        {
          *v23 = 0x203A544345534942LL;
          v25 = (_BYTE *)(*(_QWORD *)(v20 + 32) + 8LL);
          *(_QWORD *)(v24 + 32) = v25;
        }
        v26 = *(_QWORD *)(v24 + 24);
        v27 = v26 - (_QWORD)v25;
        if ( v21 > v26 - (__int64)v25 )
        {
          v24 = sub_CB6200(v24, v22, v21);
          v25 = *(_BYTE **)(v24 + 32);
          v27 = *(_QWORD *)(v24 + 24) - (_QWORD)v25;
        }
        else if ( v21 )
        {
          if ( (_DWORD)v21 )
          {
            v37 = 0;
            do
            {
              v38 = v37++;
              v25[v38] = v22[v38];
            }
            while ( v37 < (unsigned int)v21 );
            v26 = *(_QWORD *)(v24 + 24);
          }
          v25 = (_BYTE *)(v21 + *(_QWORD *)(v24 + 32));
          *(_QWORD *)(v24 + 32) = v25;
          v27 = v26 - (_QWORD)v25;
        }
        if ( v27 <= 0xC )
        {
          v24 = sub_CB6200(v24, "running pass ", 13);
          v28 = *(_BYTE **)(v24 + 32);
          if ( v28 != *(_BYTE **)(v24 + 24) )
            goto LABEL_30;
        }
        else
        {
          qmemcpy(v25, "running pass ", 13);
          v28 = (_BYTE *)(*(_QWORD *)(v24 + 32) + 13LL);
          *(_QWORD *)(v24 + 32) = v28;
          if ( v28 != *(_BYTE **)(v24 + 24) )
          {
LABEL_30:
            *v28 = 40;
            ++*(_QWORD *)(v24 + 32);
            goto LABEL_31;
          }
        }
        v24 = sub_CB6200(v24, "(", 1);
LABEL_31:
        v29 = sub_CB59F0(v24, v9);
        v30 = *(_WORD **)(v29 + 32);
        v31 = v29;
        if ( *(_QWORD *)(v29 + 24) - (_QWORD)v30 <= 1u )
        {
          v42 = sub_CB6200(v29, ") ", 2);
          v32 = *(_DWORD **)(v42 + 32);
          v31 = v42;
        }
        else
        {
          *v30 = 8233;
          v32 = (_DWORD *)(*(_QWORD *)(v29 + 32) + 2LL);
          *(_QWORD *)(v29 + 32) = v32;
        }
        v33 = *(_QWORD *)(v31 + 24) - (_QWORD)v32;
        if ( v33 < a3 )
        {
          v41 = sub_CB6200(v31, v5, a3);
          v32 = *(_DWORD **)(v41 + 32);
          v31 = v41;
          v33 = *(_QWORD *)(v41 + 24) - (_QWORD)v32;
        }
        else if ( a3 )
        {
          memcpy(v32, v5, a3);
          v36 = *(_QWORD *)(v31 + 24);
          v32 = (_DWORD *)(a3 + *(_QWORD *)(v31 + 32));
          *(_QWORD *)(v31 + 32) = v32;
          v33 = v36 - (_QWORD)v32;
        }
        if ( v33 <= 3 )
        {
          v39 = sub_CB6200(v31, " on ", 4);
          v34 = *(_BYTE **)(v39 + 32);
          v31 = v39;
        }
        else
        {
          *v32 = 544108320;
          v34 = (_BYTE *)(*(_QWORD *)(v31 + 32) + 4LL);
          *(_QWORD *)(v31 + 32) = v34;
        }
        v35 = *(_BYTE **)(v31 + 24);
        if ( a5 > v35 - v34 )
        {
          v31 = sub_CB6200(v31, a4, a5);
          v34 = *(_BYTE **)(v31 + 32);
          if ( *(_BYTE **)(v31 + 24) != v34 )
          {
LABEL_42:
            *v34 = 10;
            ++*(_QWORD *)(v31 + 32);
            return v13;
          }
        }
        else
        {
          if ( a5 )
          {
            memcpy(v34, a4, a5);
            v35 = *(_BYTE **)(v31 + 24);
            v34 = (_BYTE *)(a5 + *(_QWORD *)(v31 + 32));
            *(_QWORD *)(v31 + 32) = v34;
          }
          if ( v35 != v34 )
            goto LABEL_42;
        }
        sub_CB6200(v31, "\n", 1);
        return v13;
      }
    }
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 64);
    if ( v16 )
    {
      a2 = (_DWORD *)(a1 + 56);
      do
      {
        while ( 1 )
        {
          v17 = *(_QWORD *)(v16 + 16);
          v18 = *(_QWORD *)(v16 + 24);
          if ( v9 <= *(_DWORD *)(v16 + 32) )
            break;
          v16 = *(_QWORD *)(v16 + 24);
          if ( !v18 )
            goto LABEL_17;
        }
        a2 = (_DWORD *)v16;
        v16 = *(_QWORD *)(v16 + 16);
      }
      while ( v17 );
LABEL_17:
      if ( (_DWORD *)(a1 + 56) != a2 && v9 >= a2[8] )
        goto LABEL_9;
    }
  }
LABEL_3:
  v12 = (unsigned __int8)qword_4F82028;
  if ( (*(_DWORD *)(a1 + 8) & 0x7FFFFFFF) != 0x7FFFFFFF )
  {
    if ( v9 > *(_DWORD *)(a1 + 8) )
      goto LABEL_10;
    v13 = 1;
    if ( !(_BYTE)qword_4F82028 )
      return v13;
LABEL_5:
    v13 = 1;
    v14 = byte_3F871B3;
    goto LABEL_24;
  }
  v13 = 1;
  if ( (_BYTE)qword_4F82028 )
    goto LABEL_5;
  return v13;
}
