// Function: sub_107A6B0
// Address: 0x107a6b0
//
void __fastcall sub_107A6B0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v7; // rbx
  size_t v8; // r13
  size_t v9; // r15
  __int64 v10; // r14
  char v11; // si
  char v12; // al
  char *v13; // rax
  __int64 v14; // r15
  unsigned __int64 v15; // rcx
  char *v16; // rdi
  size_t v17; // r13
  size_t v18; // r14
  char v19; // si
  char v20; // al
  __int64 v21; // r8
  unsigned __int64 v22; // rax
  char *v23; // rdi
  char v24; // si
  char v25; // dl
  __int64 v26; // rdi
  char *v27; // rax
  char v28; // dl
  __int64 v29; // rdi
  char *v30; // rax
  int v31; // edx
  __int64 v32; // rdi
  _BYTE *v33; // rax
  unsigned __int64 v34; // r13
  __int64 v35; // r15
  char v36; // si
  char v37; // al
  char *v38; // rax
  unsigned __int64 v39; // r13
  __int64 v40; // r15
  char v41; // si
  char v42; // al
  char *v43; // rax
  __int64 v44; // rdi
  _BYTE *v45; // rax
  __int64 *v46; // rax
  __int64 *v47; // rax
  unsigned __int64 v48; // [rsp+8h] [rbp-78h]
  __int64 v49; // [rsp+10h] [rbp-70h]
  __int64 v51; // [rsp+20h] [rbp-60h]
  unsigned __int8 *src; // [rsp+28h] [rbp-58h]
  unsigned __int8 *srca; // [rsp+28h] [rbp-58h]
  __int64 v54[10]; // [rsp+30h] [rbp-50h] BYREF

  if ( a3 )
  {
    v7 = a2;
    v48 = (unsigned __int64)(a4 + 0xFFFF) >> 16;
    sub_1079610(a1, (__int64)v54, 2);
    sub_107A5C0(a3, **(_QWORD **)(a1 + 104), 0);
    v51 = a2 + 80 * a3;
    if ( a2 == v51 )
      goto LABEL_30;
LABEL_3:
    v8 = *(_QWORD *)(v7 + 8);
    src = *(unsigned __int8 **)v7;
    v9 = v8;
    v10 = **(_QWORD **)(a1 + 104);
    do
    {
      while ( 1 )
      {
        v11 = v9 & 0x7F;
        v12 = v9 & 0x7F | 0x80;
        v9 >>= 7;
        if ( v9 )
          v11 = v12;
        v13 = *(char **)(v10 + 32);
        if ( (unsigned __int64)v13 >= *(_QWORD *)(v10 + 24) )
          break;
        *(_QWORD *)(v10 + 32) = v13 + 1;
        *v13 = v11;
        if ( !v9 )
          goto LABEL_9;
      }
      sub_CB5D20(v10, v11);
    }
    while ( v9 );
LABEL_9:
    v14 = **(_QWORD **)(a1 + 104);
    v15 = *(_QWORD *)(v14 + 24);
    v16 = *(char **)(v14 + 32);
    if ( v8 > v15 - (unsigned __int64)v16 )
    {
      sub_CB6200(**(_QWORD **)(a1 + 104), src, v8);
      v46 = *(__int64 **)(a1 + 104);
      v14 = *v46;
      v16 = *(char **)(*v46 + 32);
      v15 = *(_QWORD *)(*v46 + 24);
    }
    else if ( v8 )
    {
      memcpy(v16, src, v8);
      *(_QWORD *)(v14 + 32) += v8;
      v47 = *(__int64 **)(a1 + 104);
      v14 = *v47;
      v16 = *(char **)(*v47 + 32);
      v15 = *(_QWORD *)(*v47 + 24);
    }
    v17 = *(_QWORD *)(v7 + 24);
    srca = *(unsigned __int8 **)(v7 + 16);
    v18 = v17;
    while ( 1 )
    {
      v19 = v18 & 0x7F;
      v20 = v18 & 0x7F | 0x80;
      v18 >>= 7;
      if ( v18 )
        v19 = v20;
      if ( v15 > (unsigned __int64)v16 )
      {
        *(_QWORD *)(v14 + 32) = v16 + 1;
        *v16 = v19;
        if ( !v18 )
          goto LABEL_19;
      }
      else
      {
        sub_CB5D20(v14, v19);
        if ( !v18 )
        {
LABEL_19:
          v21 = **(_QWORD **)(a1 + 104);
          v22 = *(_QWORD *)(v21 + 24);
          v23 = *(char **)(v21 + 32);
          if ( v22 - (unsigned __int64)v23 < v17 )
          {
            sub_CB6200(**(_QWORD **)(a1 + 104), srca, v17);
            v24 = *(_BYTE *)(v7 + 32);
            v21 = **(_QWORD **)(a1 + 104);
            v23 = *(char **)(v21 + 32);
            if ( *(_QWORD *)(v21 + 24) > (unsigned __int64)v23 )
              goto LABEL_23;
          }
          else
          {
            if ( v17 )
            {
              v49 = **(_QWORD **)(a1 + 104);
              memcpy(v23, srca, v17);
              *(_QWORD *)(v49 + 32) += v17;
              v21 = **(_QWORD **)(a1 + 104);
              v23 = *(char **)(v21 + 32);
              v22 = *(_QWORD *)(v21 + 24);
            }
            v24 = *(_BYTE *)(v7 + 32);
            if ( v22 > (unsigned __int64)v23 )
            {
LABEL_23:
              *(_QWORD *)(v21 + 32) = v23 + 1;
              *v23 = v24;
              goto LABEL_24;
            }
          }
          sub_CB5D20(v21, v24);
LABEL_24:
          switch ( *(_BYTE *)(v7 + 32) )
          {
            case 0:
              v39 = *(unsigned int *)(v7 + 40);
              v40 = **(_QWORD **)(a1 + 104);
              do
              {
                v41 = v39 & 0x7F;
                v42 = v39 & 0x7F | 0x80;
                v39 >>= 7;
                if ( v39 )
                  v41 = v42;
                v43 = *(char **)(v40 + 32);
                if ( (unsigned __int64)v43 < *(_QWORD *)(v40 + 24) )
                {
                  *(_QWORD *)(v40 + 32) = v43 + 1;
                  *v43 = v41;
                }
                else
                {
                  sub_CB5D20(v40, v41);
                }
              }
              while ( v39 );
              goto LABEL_29;
            case 1:
              v31 = *(_DWORD *)(v7 + 40);
              v32 = **(_QWORD **)(a1 + 104);
              v33 = *(_BYTE **)(v32 + 32);
              if ( (unsigned __int64)v33 >= *(_QWORD *)(v32 + 24) )
              {
                sub_CB5D20(v32, v31);
              }
              else
              {
                *(_QWORD *)(v32 + 32) = v33 + 1;
                *v33 = v31;
              }
              v34 = *(unsigned __int8 *)(v7 + 48);
              v35 = **(_QWORD **)(a1 + 104);
              break;
            case 2:
              sub_107A5C0(*(unsigned __int8 *)(v7 + 40), **(_QWORD **)(a1 + 104), 0);
              sub_107A5C0(v48, **(_QWORD **)(a1 + 104), 0);
              goto LABEL_29;
            case 3:
              v25 = *(_BYTE *)(v7 + 40);
              v26 = **(_QWORD **)(a1 + 104);
              v27 = *(char **)(v26 + 32);
              if ( (unsigned __int64)v27 >= *(_QWORD *)(v26 + 24) )
              {
                sub_CB5D20(v26, v25);
              }
              else
              {
                *(_QWORD *)(v26 + 32) = v27 + 1;
                *v27 = v25;
              }
              v28 = *(_BYTE *)(v7 + 41);
              v29 = **(_QWORD **)(a1 + 104);
              v30 = *(char **)(v29 + 32);
              if ( (unsigned __int64)v30 >= *(_QWORD *)(v29 + 24) )
              {
                sub_CB5D20(v29, v28);
              }
              else
              {
                *(_QWORD *)(v29 + 32) = v30 + 1;
                *v30 = v28;
              }
              goto LABEL_29;
            case 4:
              v44 = **(_QWORD **)(a1 + 104);
              v45 = *(_BYTE **)(v44 + 32);
              if ( (unsigned __int64)v45 >= *(_QWORD *)(v44 + 24) )
              {
                sub_CB5D20(v44, 0);
              }
              else
              {
                *(_QWORD *)(v44 + 32) = v45 + 1;
                *v45 = 0;
              }
              sub_107A5C0(*(unsigned int *)(v7 + 40), **(_QWORD **)(a1 + 104), 0);
              goto LABEL_29;
            default:
              BUG();
          }
          do
          {
            while ( 1 )
            {
              v36 = v34 & 0x7F;
              v37 = v34 & 0x7F | 0x80;
              v34 >>= 7;
              if ( v34 )
                v36 = v37;
              v38 = *(char **)(v35 + 32);
              if ( (unsigned __int64)v38 >= *(_QWORD *)(v35 + 24) )
                break;
              *(_QWORD *)(v35 + 32) = v38 + 1;
              *v38 = v36;
              if ( !v34 )
                goto LABEL_40;
            }
            sub_CB5D20(v35, v36);
          }
          while ( v34 );
LABEL_40:
          sub_107A5C0(a5, **(_QWORD **)(a1 + 104), 0);
LABEL_29:
          v7 += 80;
          if ( v51 == v7 )
          {
LABEL_30:
            sub_1077B30(a1, v54);
            return;
          }
          goto LABEL_3;
        }
      }
      v16 = *(char **)(v14 + 32);
      v15 = *(_QWORD *)(v14 + 24);
    }
  }
}
