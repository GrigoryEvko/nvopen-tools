// Function: sub_3027B30
// Address: 0x3027b30
//
void __fastcall sub_3027B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r15d
  __int64 v8; // rax
  int *v9; // rax
  int v10; // ebx
  unsigned int v11; // r14d
  __int64 v12; // rax
  _WORD *v13; // rdx
  int v14; // ebx
  _BYTE *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  _BYTE *v18; // rax
  _WORD *v19; // rdx
  unsigned int v20; // [rsp+18h] [rbp-A8h]
  int v21; // [rsp+1Ch] [rbp-A4h]
  unsigned __int64 v22; // [rsp+20h] [rbp-A0h]
  unsigned __int8 *v23; // [rsp+30h] [rbp-90h] BYREF
  size_t v24; // [rsp+38h] [rbp-88h]
  _BYTE v25[16]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v26[14]; // [rsp+50h] [rbp-70h] BYREF

  v7 = *(_DWORD *)a1;
  v21 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 168) + 208LL) + 8LL);
  if ( !*(_DWORD *)(a1 + 72) )
  {
    while ( v7 && !*(_BYTE *)(*(_QWORD *)(a1 + 8) + v7 - 1) )
      --v7;
  }
  v8 = *(unsigned int *)(a1 + 40);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 4u, a5, a6);
    v8 = *(unsigned int *)(a1 + 40);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * v8) = v7;
  v9 = *(int **)(a1 + 32);
  ++*(_DWORD *)(a1 + 40);
  v10 = *v9;
  if ( v7 )
  {
    v20 = 0;
    v11 = 0;
LABEL_6:
    if ( v11 == v10 )
    {
      while ( 1 )
      {
        v25[0] = 0;
        v26[5] = 0x100000000LL;
        v26[6] = &v23;
        v23 = v25;
        v26[0] = &unk_49DD210;
        v24 = 0;
        memset(&v26[1], 0, 32);
        sub_CB5980((__int64)v26, 0, 0, 0);
        v14 = 0;
        sub_30278D0(a1, v20, (__int64)v26);
        if ( v21 )
        {
          while ( 1 )
          {
            sub_C7F500(a2, 255LL << (8 * (unsigned __int8)v14), 2, v22, 0);
            v15 = *(_BYTE **)(a2 + 32);
            if ( *(_BYTE **)(a2 + 24) == v15 )
            {
              v16 = sub_CB6200(a2, (unsigned __int8 *)"(", 1u);
            }
            else
            {
              *v15 = 40;
              v16 = a2;
              ++*(_QWORD *)(a2 + 32);
            }
            v17 = sub_CB6200(v16, v23, v24);
            v18 = *(_BYTE **)(v17 + 32);
            if ( *(_BYTE **)(v17 + 24) == v18 )
            {
              ++v14;
              sub_CB6200(v17, (unsigned __int8 *)")", 1u);
              if ( v21 == v14 )
                break;
            }
            else
            {
              *v18 = 41;
              ++v14;
              ++*(_QWORD *)(v17 + 32);
              if ( v21 == v14 )
                break;
            }
            v19 = *(_WORD **)(a2 + 32);
            if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 <= 1u )
            {
              sub_CB6200(a2, (unsigned __int8 *)", ", 2u);
            }
            else
            {
              *v19 = 8236;
              *(_QWORD *)(a2 + 32) += 2LL;
            }
          }
        }
        ++v20;
        v11 += v21;
        v10 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * v20);
        v26[0] = &unk_49DD210;
        sub_CB5840((__int64)v26);
        if ( v23 == v25 )
          break;
        j_j___libc_free_0((unsigned __int64)v23);
        if ( v7 <= v11 )
          return;
LABEL_9:
        if ( !v11 )
          goto LABEL_6;
        v13 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v13 <= 1u )
        {
          sub_CB6200(a2, (unsigned __int8 *)", ", 2u);
          goto LABEL_6;
        }
        *v13 = 8236;
        *(_QWORD *)(a2 + 32) += 2LL;
        if ( v11 != v10 )
          goto LABEL_7;
      }
    }
    else
    {
LABEL_7:
      v12 = v11++;
      sub_CB59D0(a2, *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + v12));
    }
    if ( v7 > v11 )
      goto LABEL_9;
  }
}
