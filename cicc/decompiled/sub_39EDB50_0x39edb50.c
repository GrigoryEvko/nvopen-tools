// Function: sub_39EDB50
// Address: 0x39edb50
//
__int64 __fastcall sub_39EDB50(__int64 a1, unsigned int a2)
{
  __int64 v3; // rdi
  void *v4; // rdx
  __int64 v5; // rdi
  _BYTE *v6; // rax

  v3 = *(_QWORD *)(a1 + 272);
  v4 = *(void **)(v3 + 24);
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 0xCu )
  {
    v3 = sub_16E7EE0(v3, "\t.cv_func_id ", 0xDu);
  }
  else
  {
    qmemcpy(v4, "\t.cv_func_id ", 13);
    *(_QWORD *)(v3 + 24) += 13LL;
  }
  v5 = sub_16E7A90(v3, a2);
  v6 = *(_BYTE **)(v5 + 24);
  if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 16) )
  {
    sub_16E7DE0(v5, 10);
  }
  else
  {
    *(_QWORD *)(v5 + 24) = v6 + 1;
    *v6 = 10;
  }
  return sub_38DBE50(a1, a2);
}
