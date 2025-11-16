// Function: sub_107AB40
// Address: 0x107ab40
//
void __fastcall sub_107AB40(__int64 a1, unsigned __int8 *a2, unsigned __int64 a3)
{
  unsigned __int8 *v5; // rbx
  unsigned __int64 v6; // r15
  __int64 v7; // r13
  char v8; // si
  char v9; // al
  char *v10; // rax
  char v11; // si
  __int64 v12; // rdi
  char *v13; // rax
  char v14; // si
  __int64 v15; // rdi
  char *v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 *v21; // rax
  __int64 *v22; // rax
  __int64 v23; // rdi
  _BYTE *v24; // rax
  unsigned __int8 *v25; // [rsp-70h] [rbp-70h]
  __int64 v26; // [rsp-60h] [rbp-60h] BYREF
  __int64 v27[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( a3 )
  {
    v5 = a2;
    sub_1079610(a1, (__int64)v27, 6);
    sub_107A5C0(a3, **(_QWORD **)(a1 + 104), 0);
    v25 = &a2[72 * a3];
    if ( a2 != v25 )
    {
      do
      {
        v6 = v5[4];
        v7 = **(_QWORD **)(a1 + 104);
        do
        {
          while ( 1 )
          {
            v8 = v6 & 0x7F;
            v9 = v6 & 0x7F | 0x80;
            v6 >>= 7;
            if ( v6 )
              v8 = v9;
            v10 = *(char **)(v7 + 32);
            if ( (unsigned __int64)v10 >= *(_QWORD *)(v7 + 24) )
              break;
            *(_QWORD *)(v7 + 32) = v10 + 1;
            *v10 = v8;
            if ( !v6 )
              goto LABEL_9;
          }
          sub_CB5D20(v7, v8);
        }
        while ( v6 );
LABEL_9:
        v11 = v5[5];
        v12 = **(_QWORD **)(a1 + 104);
        v13 = *(char **)(v12 + 32);
        if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
        {
          sub_CB5D20(v12, v11);
        }
        else
        {
          *(_QWORD *)(v12 + 32) = v13 + 1;
          *v13 = v11;
        }
        if ( v5[8] )
          BUG();
        v14 = v5[16];
        v15 = **(_QWORD **)(a1 + 104);
        v16 = *(char **)(v15 + 32);
        if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 24) )
        {
          sub_CB5D20(v15, v14);
        }
        else
        {
          *(_QWORD *)(v15 + 32) = v16 + 1;
          *v16 = v14;
        }
        switch ( v5[4] )
        {
          case 'o':
            v23 = **(_QWORD **)(a1 + 104);
            v24 = *(_BYTE **)(v23 + 32);
            if ( (unsigned __int64)v24 >= *(_QWORD *)(v23 + 24) )
            {
              sub_CB5D20(v23, 111);
            }
            else
            {
              *(_QWORD *)(v23 + 32) = v24 + 1;
              *v24 = 111;
            }
            break;
          case '|':
            v22 = *(__int64 **)(a1 + 104);
            v26 = 0;
            sub_CB6200(*v22, (unsigned __int8 *)&v26, 8u);
            break;
          case '}':
            v21 = *(__int64 **)(a1 + 104);
            LODWORD(v26) = 0;
            sub_CB6200(*v21, (unsigned __int8 *)&v26, 4u);
            break;
          case '~':
          case '\x7F':
            v17 = **(_QWORD **)(a1 + 104);
            v18 = *(_BYTE **)(v17 + 32);
            if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 24) )
            {
              sub_CB5D20(v17, 0);
            }
            else
            {
              *(_QWORD *)(v17 + 32) = v18 + 1;
              *v18 = 0;
            }
            break;
          default:
            BUG();
        }
        v19 = **(_QWORD **)(a1 + 104);
        v20 = *(_BYTE **)(v19 + 32);
        if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 24) )
        {
          sub_CB5D20(v19, 11);
        }
        else
        {
          *(_QWORD *)(v19 + 32) = v20 + 1;
          *v20 = 11;
        }
        v5 += 72;
      }
      while ( v5 != v25 );
    }
    sub_1077B30(a1, v27);
  }
}
