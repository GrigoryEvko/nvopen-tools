// Function: sub_2FAD6C0
// Address: 0x2fad6c0
//
void __fastcall sub_2FAD6C0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // rdi
  __int64 v6; // rdi
  _BYTE *v7; // rax
  _BYTE *v8; // rax
  unsigned int v9; // r15d
  unsigned __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rax
  _WORD *v13; // rdx
  __int64 v14; // r15
  _BYTE *v15; // rax
  __int64 v16; // rdx
  _WORD *v17; // rdx
  _DWORD *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-50h]
  __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1 + 96;
  v4 = *(_QWORD *)(a1 + 104);
  if ( v4 != a1 + 96 )
  {
    do
    {
      while ( 1 )
      {
        v6 = sub_CB59D0(a2, *(unsigned int *)(v4 + 24));
        v7 = *(_BYTE **)(v6 + 32);
        if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 24) )
          break;
        *(_QWORD *)(v6 + 32) = v7 + 1;
        *v7 = 32;
        v5 = *(_QWORD *)(v4 + 16);
        if ( !v5 )
          goto LABEL_8;
LABEL_4:
        sub_2E91850(v5, a2, 1u, 0, 0, 1, 0);
LABEL_5:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v3 == v4 )
          goto LABEL_10;
      }
      sub_CB5D20(v6, 32);
      v5 = *(_QWORD *)(v4 + 16);
      if ( v5 )
        goto LABEL_4;
LABEL_8:
      v8 = *(_BYTE **)(a2 + 32);
      if ( (unsigned __int64)v8 >= *(_QWORD *)(a2 + 24) )
      {
        sub_CB5D20(a2, 10);
        goto LABEL_5;
      }
      *(_QWORD *)(a2 + 32) = v8 + 1;
      *v8 = 10;
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v3 != v4 );
  }
LABEL_10:
  v9 = *(_DWORD *)(a1 + 160);
  if ( v9 )
  {
    v10 = 0;
    v20 = v9;
    do
    {
      while ( 1 )
      {
        v18 = *(_DWORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v18 > 3u )
        {
          *v18 = 778199589;
          v11 = a2;
          *(_QWORD *)(a2 + 32) += 4LL;
        }
        else
        {
          v11 = sub_CB6200(a2, (unsigned __int8 *)"%bb.", 4u);
        }
        v12 = sub_CB59D0(v11, v10);
        v13 = *(_WORD **)(v12 + 32);
        v14 = v12;
        if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 1u )
        {
          v14 = sub_CB6200(v12, "\t[", 2u);
        }
        else
        {
          *v13 = 23305;
          *(_QWORD *)(v12 + 32) += 2LL;
        }
        v21[0] = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16 * v10);
        sub_2FAD600(v21, v14);
        v15 = *(_BYTE **)(v14 + 32);
        v16 = 16 * v10;
        if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 24) )
        {
          v19 = sub_CB5D20(v14, 59);
          v16 = 16 * v10;
          v14 = v19;
        }
        else
        {
          *(_QWORD *)(v14 + 32) = v15 + 1;
          *v15 = 59;
        }
        v21[0] = *(_QWORD *)(*(_QWORD *)(a1 + 152) + v16 + 8);
        sub_2FAD600(v21, v14);
        v17 = *(_WORD **)(v14 + 32);
        if ( *(_QWORD *)(v14 + 24) - (_QWORD)v17 <= 1u )
          break;
        ++v10;
        *v17 = 2601;
        *(_QWORD *)(v14 + 32) += 2LL;
        if ( v10 == v20 )
          return;
      }
      ++v10;
      sub_CB6200(v14, (unsigned __int8 *)")\n", 2u);
    }
    while ( v10 != v20 );
  }
}
