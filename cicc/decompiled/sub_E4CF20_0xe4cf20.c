// Function: sub_E4CF20
// Address: 0xe4cf20
//
__int64 __fastcall sub_E4CF20(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v4; // r12
  __int64 v6; // rdi
  void *v7; // rdx
  __int64 result; // rax
  __int64 *v9; // r14
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r15
  __int64 v14; // r13
  _BYTE *v15; // rax

  v4 = a2;
  v6 = *(_QWORD *)(a1 + 304);
  v7 = *(void **)(v6 + 32);
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 0xEu )
  {
    result = sub_CB6200(v6, "\t.cv_def_range\t", 0xFu);
  }
  else
  {
    qmemcpy(v7, "\t.cv_def_range\t", 15);
    result = 25959;
    *(_QWORD *)(v6 + 32) += 15LL;
  }
  v9 = &a2[2 * a3];
  if ( a2 != v9 )
  {
    do
    {
      v12 = *(_QWORD *)(a1 + 304);
      v13 = *v4;
      v14 = v4[1];
      v15 = *(_BYTE **)(v12 + 32);
      if ( (unsigned __int64)v15 < *(_QWORD *)(v12 + 24) )
      {
        *(_QWORD *)(v12 + 32) = v15 + 1;
        *v15 = 32;
      }
      else
      {
        sub_CB5D20(v12, 32);
      }
      sub_EA12C0(v13, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
      v10 = *(_QWORD *)(a1 + 304);
      v11 = *(_BYTE **)(v10 + 32);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
      {
        sub_CB5D20(v10, 32);
      }
      else
      {
        *(_QWORD *)(v10 + 32) = v11 + 1;
        *v11 = 32;
      }
      v4 += 2;
      result = sub_EA12C0(v14, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
    }
    while ( v9 != v4 );
  }
  return result;
}
