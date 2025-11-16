// Function: sub_E4E160
// Address: 0xe4e160
//
_BYTE *__fastcall sub_E4E160(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int8 a4)
{
  __int64 v6; // rdi
  _QWORD *v8; // rdx
  __int64 v9; // rdi
  _BYTE *v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __int64 v15; // rdi
  _BYTE *v16; // rax

  v6 = *(_QWORD *)(a1 + 304);
  v8 = *(_QWORD **)(v6 + 32);
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v8 <= 7u )
  {
    sub_CB6200(v6, "\t.lcomm\t", 8u);
  }
  else
  {
    *v8 = 0x96D6D6F636C2E09LL;
    *(_QWORD *)(v6 + 32) += 8LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v9 = *(_QWORD *)(a1 + 304);
  v10 = *(_BYTE **)(v9 + 32);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
  {
    v9 = sub_CB5D20(v9, 44);
  }
  else
  {
    *(_QWORD *)(v9 + 32) = v10 + 1;
    *v10 = 44;
  }
  sub_CB59D0(v9, a3);
  if ( (unsigned __int64)(1LL << a4) <= 1 )
    return sub_E4D880(a1);
  v11 = *(_DWORD *)(*(_QWORD *)(a1 + 312) + 284LL);
  if ( v11 != 2 )
  {
    if ( v11 <= 2 )
    {
      if ( !v11 )
        BUG();
      v12 = *(_QWORD *)(a1 + 304);
      v13 = *(_BYTE **)(v12 + 32);
      if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
      {
        v12 = sub_CB5D20(v12, 44);
      }
      else
      {
        *(_QWORD *)(v12 + 32) = v13 + 1;
        *v13 = 44;
      }
      sub_CB59D0(v12, 1LL << a4);
    }
    return sub_E4D880(a1);
  }
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
  sub_CB59D0(v15, a4);
  return sub_E4D880(a1);
}
