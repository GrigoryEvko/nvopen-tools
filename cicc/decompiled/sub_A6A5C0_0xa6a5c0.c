// Function: sub_A6A5C0
// Address: 0xa6a5c0
//
__int64 __fastcall sub_A6A5C0(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // rdi
  void *v4; // rdx
  _QWORD *v5; // r14
  _QWORD *v6; // rbx
  char v7; // r13
  _WORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rax
  _WORD *v13; // rdx
  __int64 v14; // r15
  _BYTE *v15; // rax
  int v16; // eax
  __int64 v17; // rdi
  _BYTE *v18; // rax

  v3 = *a1;
  v4 = *(void **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0xBu )
  {
    sub_CB6200(v3, ", summary: (", 12);
  }
  else
  {
    qmemcpy(v4, ", summary: (", 12);
    *(_QWORD *)(v3 + 32) += 12LL;
  }
  v5 = (_QWORD *)a2[1];
  v6 = (_QWORD *)*a2;
  v7 = 1;
  if ( (_QWORD *)*a2 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v9 = *a1;
        v10 = *a1;
        if ( v7 )
        {
          v7 = 0;
        }
        else
        {
          v8 = *(_WORD **)(v9 + 32);
          if ( *(_QWORD *)(v9 + 24) - (_QWORD)v8 <= 1u )
          {
            sub_CB6200(v9, ", ", 2);
          }
          else
          {
            *v8 = 8236;
            *(_QWORD *)(v9 + 32) += 2LL;
          }
          v9 = *a1;
          v10 = *a1;
        }
        v11 = *(_QWORD *)(v9 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v9 + 24) - v11) <= 8 )
        {
          v10 = sub_CB6200(v9, "(offset: ", 9);
        }
        else
        {
          *(_BYTE *)(v11 + 8) = 32;
          *(_QWORD *)v11 = 0x3A74657366666F28LL;
          *(_QWORD *)(v9 + 32) += 9LL;
        }
        v12 = sub_CB59D0(v10, *v6);
        v13 = *(_WORD **)(v12 + 32);
        if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 1u )
        {
          sub_CB6200(v12, ", ", 2);
        }
        else
        {
          *v13 = 8236;
          *(_QWORD *)(v12 + 32) += 2LL;
        }
        v14 = *a1;
        v15 = *(_BYTE **)(*a1 + 32);
        if ( *(_BYTE **)(*a1 + 24) == v15 )
        {
          v14 = sub_CB6200(*a1, "^", 1);
        }
        else
        {
          *v15 = 94;
          ++*(_QWORD *)(v14 + 32);
        }
        v16 = sub_A6A250(a1[4], *(_QWORD *)(v6[1] & 0xFFFFFFFFFFFFFFF8LL));
        sub_CB59F0(v14, v16);
        v17 = *a1;
        v18 = *(_BYTE **)(*a1 + 32);
        if ( *(_BYTE **)(*a1 + 24) == v18 )
          break;
        v6 += 2;
        *v18 = 41;
        ++*(_QWORD *)(v17 + 32);
        if ( v5 == v6 )
          return sub_904010(*a1, ")");
      }
      v6 += 2;
      sub_CB6200(v17, ")", 1);
    }
    while ( v5 != v6 );
  }
  return sub_904010(*a1, ")");
}
