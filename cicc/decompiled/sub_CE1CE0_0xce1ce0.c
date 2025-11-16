// Function: sub_CE1CE0
// Address: 0xce1ce0
//
__int64 __fastcall sub_CE1CE0(__int64 *a1, __int64 a2)
{
  _BYTE *v3; // rax
  char v4; // dl
  __int64 result; // rax
  bool v6; // bl
  __int64 v7; // r15
  __int64 v8; // r14
  void *v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // r12
  void *v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rdi

  v3 = (_BYTE *)unk_4F83008;
  v4 = 0;
  if ( unk_4F83008 != unk_4F83010 )
  {
    do
      v4 |= *v3++;
    while ( (_BYTE *)unk_4F83010 != v3 );
  }
  result = (v4 & 2) != 0;
  v6 = (v4 & 4) != 0;
  if ( (v4 & 4) != 0 )
  {
    v7 = *(_QWORD *)(a2 + 40);
    if ( (v4 & 2) == 0 )
    {
LABEL_13:
      v12 = *a1;
      v13 = *(void **)(*a1 + 32);
      if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v13 <= 0xCu )
      {
        v12 = sub_CB6200(*a1, "Module Size: ", 0xDu);
      }
      else
      {
        qmemcpy(v13, "Module Size: ", 13);
        *(_QWORD *)(v12 + 32) += 13LL;
      }
      v14 = sub_BAA3C0(v7);
      v15 = sub_CB59D0(v12, v14);
      result = *(_QWORD *)(v15 + 32);
      if ( *(_QWORD *)(v15 + 24) == result )
        return sub_CB6200(v15, (unsigned __int8 *)"\t", 1u);
      *(_BYTE *)result = 9;
      ++*(_QWORD *)(v15 + 32);
      return result;
    }
  }
  else
  {
    if ( (v4 & 2) == 0 )
      return result;
    v7 = *(_QWORD *)(a2 + 40);
  }
  v8 = *a1;
  v9 = *(void **)(*a1 + 32);
  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v9 <= 0xEu )
  {
    v8 = sub_CB6200(*a1, "Function Size: ", 0xFu);
  }
  else
  {
    qmemcpy(v9, "Function Size: ", 15);
    *(_QWORD *)(v8 + 32) += 15LL;
  }
  v10 = sub_B2BED0(a2);
  v11 = sub_CB59D0(v8, v10);
  result = *(_QWORD *)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) == result )
  {
    result = sub_CB6200(v11, (unsigned __int8 *)"\t", 1u);
  }
  else
  {
    *(_BYTE *)result = 9;
    ++*(_QWORD *)(v11 + 32);
  }
  if ( v6 )
    goto LABEL_13;
  return result;
}
