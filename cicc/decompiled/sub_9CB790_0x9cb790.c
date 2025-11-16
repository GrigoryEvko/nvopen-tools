// Function: sub_9CB790
// Address: 0x9cb790
//
bool __fastcall sub_9CB790(
        __int64 a1,
        __int64 a2,
        unsigned int *a3,
        unsigned int a4,
        __int64 *a5,
        _DWORD *a6,
        __int64 a7)
{
  __int64 v8; // r8
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // r8
  unsigned int v13; // r13d
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v24; // rax
  _QWORD *v25; // [rsp+8h] [rbp-28h]

  v8 = *a3;
  if ( (_DWORD)v8 == *(_DWORD *)(a2 + 8) )
    return 1;
  v10 = (unsigned int)(v8 + 1);
  v11 = *a3;
  *a3 = v10;
  v12 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v8);
  v13 = v12;
  if ( !*(_BYTE *)(a1 + 1832) )
  {
    if ( (unsigned int)v12 >= a4 )
      goto LABEL_4;
LABEL_11:
    v19 = a7;
    v20 = a1 + 744;
    v18 = 0;
    v21 = *(unsigned int *)(*(_QWORD *)(a1 + 744) + 32LL * v13 + 24);
    *a6 = v21;
    goto LABEL_8;
  }
  v13 = a4 - v12;
  if ( a4 >= (unsigned int)v12 )
    goto LABEL_11;
LABEL_4:
  if ( (_DWORD)v10 != *(_DWORD *)(a2 + 8) )
  {
    *a3 = v11 + 2;
    v14 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v10);
    *a6 = v14;
    v15 = sub_9CAD80((_QWORD *)a1, (unsigned int)v14);
    v18 = v15;
    if ( v15 && *(_BYTE *)(v15 + 8) == 9 )
    {
      v25 = (_QWORD *)v15;
      v24 = sub_A12C40(a1 + 808, v13, v15, v16, v17);
      v22 = sub_B9F6F0(*v25, v24);
      goto LABEL_9;
    }
    v19 = a7;
    v20 = a1 + 744;
    v21 = (unsigned int)v14;
LABEL_8:
    v22 = sub_A14C90(v20, v13, v18, v21, v19);
LABEL_9:
    *a5 = v22;
    return v22 == 0;
  }
  return 1;
}
