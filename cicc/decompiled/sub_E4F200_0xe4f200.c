// Function: sub_E4F200
// Address: 0xe4f200
//
_BYTE *__fastcall sub_E4F200(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v9; // rdi
  _QWORD *v10; // rdx
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // rdi
  _BYTE *v16; // rax
  _BYTE *result; // rax

  v9 = *(_QWORD *)(a1 + 304);
  v10 = *(_QWORD **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 7u )
  {
    sub_CB6200(v9, "\t.lcomm\t", 8u);
  }
  else
  {
    *v10 = 0x96D6D6F636C2E09LL;
    *(_QWORD *)(v9 + 32) += 8LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v11 = *(_QWORD *)(a1 + 304);
  v12 = *(_BYTE **)(v11 + 32);
  if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
  {
    v11 = sub_CB5D20(v11, 44);
  }
  else
  {
    *(_QWORD *)(v11 + 32) = v12 + 1;
    *v12 = 44;
  }
  v13 = sub_CB59D0(v11, a3);
  v14 = *(_BYTE **)(v13 + 32);
  if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 24) )
  {
    sub_CB5D20(v13, 44);
  }
  else
  {
    *(_QWORD *)(v13 + 32) = v14 + 1;
    *v14 = 44;
  }
  sub_EA12C0(a4, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v15 = *(_QWORD *)(a1 + 304);
  v16 = *(_BYTE **)(v15 + 32);
  if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 24) )
  {
    v15 = sub_CB5D20(v15, 44);
  }
  else
  {
    *(_QWORD *)(v15 + 32) = v16 + 1;
    *v16 = 44;
  }
  sub_CB59D0(v15, a5);
  result = sub_E4D880(a1);
  if ( *(_BYTE *)(a4 + 72) )
    return sub_E4E8B0(a1, a4, *(char **)(a4 + 56), *(_QWORD *)(a4 + 64));
  return result;
}
