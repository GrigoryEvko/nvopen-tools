// Function: sub_2F44C20
// Address: 0x2f44c20
//
unsigned __int64 __fastcall sub_2F44C20(__int64 a1, __int64 a2)
{
  bool v2; // r13
  __int64 v5; // rax
  _BOOL4 v6; // eax
  void *v7; // rdx
  char v8; // r14
  unsigned __int64 result; // rax
  _BYTE *v10; // rax
  void *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r15
  void *v14; // rdi
  unsigned __int64 v15; // r13
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax

  v2 = 0;
  if ( *(_QWORD *)(a1 + 40) == 3 )
  {
    v5 = *(_QWORD *)(a1 + 32);
    v6 = *(_WORD *)v5 != 27745 || *(_BYTE *)(v5 + 2) != 108;
    v2 = !v6;
  }
  v7 = *(void **)(a2 + 32);
  v8 = *(_BYTE *)(a1 + 48);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 0xBu )
  {
    result = sub_CB6200(a2, "regallocfast", 0xCu);
  }
  else
  {
    result = 0x636F6C6C61676572LL;
    qmemcpy(v7, "regallocfast", 12);
    *(_QWORD *)(a2 + 32) += 12LL;
  }
  if ( !v8 || !v2 )
  {
    v10 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(a2 + 24) )
    {
      sub_CB5D20(a2, 60);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = v10 + 1;
      *v10 = 60;
    }
    if ( !v2 )
    {
      v12 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v12) <= 6 )
      {
        v17 = sub_CB6200(a2, "filter=", 7u);
        v14 = *(void **)(v17 + 32);
        v13 = v17;
      }
      else
      {
        *(_DWORD *)v12 = 1953261926;
        v13 = a2;
        *(_WORD *)(v12 + 4) = 29285;
        *(_BYTE *)(v12 + 6) = 61;
        v14 = (void *)(*(_QWORD *)(a2 + 32) + 7LL);
        *(_QWORD *)(a2 + 32) = v14;
      }
      v15 = *(_QWORD *)(a1 + 40);
      v16 = *(unsigned __int8 **)(a1 + 32);
      if ( v15 > *(_QWORD *)(v13 + 24) - (_QWORD)v14 )
      {
        sub_CB6200(v13, v16, *(_QWORD *)(a1 + 40));
      }
      else if ( v15 )
      {
        memcpy(v14, v16, *(_QWORD *)(a1 + 40));
        *(_QWORD *)(v13 + 32) += v15;
      }
      result = *(_QWORD *)(a2 + 32);
      if ( v8 )
        goto LABEL_23;
      if ( result >= *(_QWORD *)(a2 + 24) )
      {
        sub_CB5D20(a2, 59);
      }
      else
      {
        *(_QWORD *)(a2 + 32) = result + 1;
        *(_BYTE *)result = 59;
      }
    }
    v11 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v11 <= 0xDu )
    {
      sub_CB6200(a2, "no-clear-vregs", 0xEu);
      result = *(_QWORD *)(a2 + 32);
    }
    else
    {
      qmemcpy(v11, "no-clear-vregs", 14);
      result = *(_QWORD *)(a2 + 32) + 14LL;
      *(_QWORD *)(a2 + 32) = result;
    }
LABEL_23:
    if ( result >= *(_QWORD *)(a2 + 24) )
    {
      return sub_CB5D20(a2, 62);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = result + 1;
      *(_BYTE *)result = 62;
    }
  }
  return result;
}
