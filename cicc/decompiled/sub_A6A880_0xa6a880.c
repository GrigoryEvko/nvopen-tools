// Function: sub_A6A880
// Address: 0xa6a880
//
__int64 __fastcall sub_A6A880(__int64 *a1, unsigned __int64 a2, unsigned __int64 a3)
{
  char v4; // r13
  __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // rdx
  _QWORD *v9; // r15
  _WORD *v10; // rdx
  __int64 v11; // rdi
  void *v12; // rdx
  int v13; // eax
  __int64 v14; // rdi
  int v15; // ecx
  _BYTE *v16; // rax
  __int64 v17; // rdi
  void *v18; // rdx
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 *v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 result; // rax
  int v26; // [rsp+8h] [rbp-48h]
  __int64 *v27; // [rsp+8h] [rbp-48h]
  unsigned __int64 v28[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = 1;
  v6 = a1[2];
  v28[0] = a2;
  v28[1] = a3;
  v7 = sub_A568D0(v6 + 208, v28);
  v9 = v8;
  if ( v8 == v7 )
  {
    sub_904010(*a1, "vFuncId: (");
    v23 = sub_904010(*a1, "guid: ");
    sub_CB59D0(v23, v28[0]);
    v24 = sub_904010(*a1, ", offset: ");
    sub_CB59D0(v24, a3);
    return sub_904010(*a1, ")");
  }
  else
  {
    do
    {
      while ( 1 )
      {
        v11 = *a1;
        v21 = v7 + 5;
        if ( v4 )
        {
          v4 = 0;
        }
        else
        {
          v10 = *(_WORD **)(v11 + 32);
          if ( *(_QWORD *)(v11 + 24) - (_QWORD)v10 <= 1u )
          {
            sub_CB6200(v11, ", ", 2);
            v21 = v7 + 5;
          }
          else
          {
            *v10 = 8236;
            *(_QWORD *)(v11 + 32) += 2LL;
          }
          v11 = *a1;
        }
        v12 = *(void **)(v11 + 32);
        if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 9u )
        {
          v27 = v21;
          sub_CB6200(v11, "vFuncId: (", 10);
          v21 = v27;
        }
        else
        {
          qmemcpy(v12, "vFuncId: (", 10);
          *(_QWORD *)(v11 + 32) += 10LL;
        }
        v13 = sub_A6A800(a1[4], *v21, v21[1]);
        v14 = *a1;
        v15 = v13;
        v16 = *(_BYTE **)(*a1 + 32);
        if ( *(_BYTE **)(*a1 + 24) == v16 )
        {
          v26 = v15;
          v22 = sub_CB6200(v14, "^", 1);
          v15 = v26;
          v14 = v22;
        }
        else
        {
          *v16 = 94;
          ++*(_QWORD *)(v14 + 32);
        }
        sub_CB59F0(v14, v15);
        v17 = *a1;
        v18 = *(void **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v18 <= 9u )
        {
          v17 = sub_CB6200(v17, ", offset: ", 10);
        }
        else
        {
          qmemcpy(v18, ", offset: ", 10);
          *(_QWORD *)(v17 + 32) += 10LL;
        }
        sub_CB59D0(v17, a3);
        v19 = *a1;
        v20 = *(_BYTE **)(*a1 + 32);
        if ( *(_BYTE **)(*a1 + 24) == v20 )
          break;
        *v20 = 41;
        ++*(_QWORD *)(v19 + 32);
        result = sub_220EF30(v7);
        v7 = (_QWORD *)result;
        if ( v9 == (_QWORD *)result )
          return result;
      }
      sub_CB6200(v19, ")", 1);
      result = sub_220EF30(v7);
      v7 = (_QWORD *)result;
    }
    while ( v9 != (_QWORD *)result );
  }
  return result;
}
