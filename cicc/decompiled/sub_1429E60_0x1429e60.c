// Function: sub_1429E60
// Address: 0x1429e60
//
void __fastcall sub_1429E60(__int64 a1, const char *a2, size_t a3, const char *a4, size_t a5, unsigned int a6)
{
  __int64 v8; // r12
  __int64 v10; // rdx
  void *v11; // rdi
  _BYTE *v12; // r15
  __int64 v13; // rdi
  _BYTE *v14; // r8
  _BYTE *v15; // rax
  __int64 v16; // rax
  const char *v17; // [rsp-40h] [rbp-40h]

  if ( !a3 )
    return;
  v8 = a1;
  v10 = *(_QWORD *)(a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 16) - v10) <= 5 )
  {
    v17 = a4;
    sub_16E7EE0(a1, " from ", 6);
    v11 = *(void **)(a1 + 24);
    a4 = v17;
    if ( !a5 )
      goto LABEL_4;
  }
  else
  {
    *(_DWORD *)v10 = 1869768224;
    *(_WORD *)(v10 + 4) = 8301;
    v11 = (void *)(*(_QWORD *)(a1 + 24) + 6LL);
    *(_QWORD *)(v8 + 24) = v11;
    if ( !a5 )
      goto LABEL_4;
  }
  if ( a5 > *(_QWORD *)(v8 + 16) - (_QWORD)v11 )
  {
    v16 = sub_16E7EE0(v8, a4, a5);
    v14 = *(_BYTE **)(v16 + 24);
    v13 = v16;
  }
  else
  {
    memcpy(v11, a4, a5);
    v12 = (_BYTE *)(*(_QWORD *)(v8 + 24) + a5);
    v13 = v8;
    *(_QWORD *)(v8 + 24) = v12;
    v14 = v12;
  }
  if ( v14 == *(_BYTE **)(v13 + 16) )
  {
    sub_16E7EE0(v13, "/", 1);
  }
  else
  {
    *v14 = 47;
    ++*(_QWORD *)(v13 + 24);
  }
  v11 = *(void **)(v8 + 24);
LABEL_4:
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v11 < a3 )
  {
    sub_16E7EE0(v8, a2, a3);
  }
  else
  {
    memcpy(v11, a2, a3);
    *(_QWORD *)(v8 + 24) += a3;
  }
  if ( a6 )
  {
    v15 = *(_BYTE **)(v8 + 24);
    if ( *(_BYTE **)(v8 + 16) == v15 )
    {
      v8 = sub_16E7EE0(v8, ":", 1);
    }
    else
    {
      *v15 = 58;
      ++*(_QWORD *)(v8 + 24);
    }
    sub_16E7A90(v8, a6);
  }
}
