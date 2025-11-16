// Function: sub_1C28B20
// Address: 0x1c28b20
//
__int64 __fastcall sub_1C28B20(__int64 *a1, __int64 a2)
{
  _BYTE *v3; // rax
  _BYTE *v4; // rcx
  char i; // dl
  __int64 result; // rax
  bool v7; // bl
  __int64 v8; // r15
  __int64 v9; // r14
  void *v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rdi
  __int64 v13; // r12
  void *v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rdi

  v3 = (_BYTE *)qword_4F9E580[20];
  v4 = (_BYTE *)qword_4F9E580[21];
  for ( i = 0; v4 != v3; ++v3 )
    i |= *v3;
  result = (i & 2) != 0;
  v7 = (i & 4) != 0;
  if ( (i & 4) != 0 )
  {
    v8 = *(_QWORD *)(a2 + 40);
    if ( (i & 2) == 0 )
    {
LABEL_13:
      v13 = *a1;
      v14 = *(void **)(*a1 + 24);
      if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v14 <= 0xCu )
      {
        v13 = sub_16E7EE0(*a1, "Module Size: ", 0xDu);
      }
      else
      {
        qmemcpy(v14, "Module Size: ", 13);
        *(_QWORD *)(v13 + 24) += 13LL;
      }
      v15 = sub_1633B40(v8);
      v16 = sub_16E7A90(v13, v15);
      result = *(_QWORD *)(v16 + 24);
      if ( *(_QWORD *)(v16 + 16) == result )
        return sub_16E7EE0(v16, "\t", 1u);
      *(_BYTE *)result = 9;
      ++*(_QWORD *)(v16 + 24);
      return result;
    }
  }
  else
  {
    if ( (i & 2) == 0 )
      return result;
    v8 = *(_QWORD *)(a2 + 40);
  }
  v9 = *a1;
  v10 = *(void **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v10 <= 0xEu )
  {
    v9 = sub_16E7EE0(*a1, "Function Size: ", 0xFu);
  }
  else
  {
    qmemcpy(v10, "Function Size: ", 15);
    *(_QWORD *)(v9 + 24) += 15LL;
  }
  v11 = sub_15E0540(a2);
  v12 = sub_16E7A90(v9, v11);
  result = *(_QWORD *)(v12 + 24);
  if ( *(_QWORD *)(v12 + 16) == result )
  {
    result = sub_16E7EE0(v12, "\t", 1u);
  }
  else
  {
    *(_BYTE *)result = 9;
    ++*(_QWORD *)(v12 + 24);
  }
  if ( v7 )
    goto LABEL_13;
  return result;
}
