// Function: sub_A53560
// Address: 0xa53560
//
void *__fastcall sub_A53560(_QWORD *a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rdi
  __int64 v5; // rdx
  unsigned __int16 v6; // ax
  const void *v7; // rax
  size_t v8; // rdx
  size_t v9; // r12
  __int64 v10; // r13
  void *v11; // rdi
  void *result; // rax
  __int64 v13; // r12
  unsigned __int16 v14; // ax

  v3 = *((_BYTE *)a1 + 8) == 0;
  v4 = *a1;
  if ( v3 )
    v4 = sub_904010(v4, (const char *)a1[2]);
  else
    *((_BYTE *)a1 + 8) = 0;
  v5 = *(_QWORD *)(v4 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 24) - v5) <= 4 )
  {
    sub_CB6200(v4, "tag: ", 5);
  }
  else
  {
    *(_DWORD *)v5 = 979853684;
    *(_BYTE *)(v5 + 4) = 32;
    *(_QWORD *)(v4 + 32) += 5LL;
  }
  v6 = sub_AF18C0(a2);
  v7 = (const void *)sub_E02B90(v6);
  v9 = v8;
  if ( v8 )
  {
    v10 = *a1;
    v11 = *(void **)(*a1 + 32LL);
    if ( *(_QWORD *)(*a1 + 24LL) - (_QWORD)v11 < v8 )
    {
      return (void *)sub_CB6200(*a1, v7, v8);
    }
    else
    {
      result = memcpy(v11, v7, v8);
      *(_QWORD *)(v10 + 32) += v9;
    }
  }
  else
  {
    v13 = *a1;
    v14 = sub_AF18C0(a2);
    return (void *)sub_CB59F0(v13, v14);
  }
  return result;
}
