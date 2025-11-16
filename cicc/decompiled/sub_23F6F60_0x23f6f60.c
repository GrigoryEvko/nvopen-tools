// Function: sub_23F6F60
// Address: 0x23f6f60
//
__int64 __fastcall sub_23F6F60(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  _BYTE *v10; // rax
  size_t v11; // r13
  _DWORD *v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 result; // rax
  _BYTE *v16; // rax
  __int64 v17; // rax

  v6 = a3(a4, "BoundsCheckingPass]", 18);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_BYTE **)(a2 + 24);
  v11 = v7;
  if ( v10 - v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v10 = *(_BYTE **)(a2 + 24);
    v8 = *(_BYTE **)(a2 + 32);
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v16 = *(_BYTE **)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( v8 != v16 )
      goto LABEL_4;
    goto LABEL_19;
  }
  if ( v8 != v10 )
  {
LABEL_4:
    *v8 = 60;
    v12 = (_DWORD *)(*(_QWORD *)(a2 + 32) + 1LL);
    *(_QWORD *)(a2 + 32) = v12;
    goto LABEL_5;
  }
LABEL_19:
  sub_CB6200(a2, "<", 1u);
  v12 = *(_DWORD **)(a2 + 32);
LABEL_5:
  v13 = *(_QWORD *)(a2 + 24);
  v14 = v13 - (_QWORD)v12;
  if ( a1[2] )
  {
    if ( *a1 )
    {
      if ( v14 <= 3 )
      {
        sub_CB6200(a2, "min-", 4u);
        v12 = *(_DWORD **)(a2 + 32);
        v13 = *(_QWORD *)(a2 + 24);
      }
      else
      {
        *v12 = 762210669;
        v12 = (_DWORD *)(*(_QWORD *)(a2 + 32) + 4LL);
        v13 = *(_QWORD *)(a2 + 24);
        *(_QWORD *)(a2 + 32) = v12;
      }
    }
    if ( (unsigned __int64)(v13 - (_QWORD)v12) <= 1 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"rt", 2u);
      result = *(_QWORD *)(a2 + 32);
      if ( a1[1] )
        goto LABEL_12;
    }
    else
    {
      *(_WORD *)v12 = 29810;
      result = *(_QWORD *)(a2 + 32) + 2LL;
      *(_QWORD *)(a2 + 32) = result;
      if ( a1[1] )
        goto LABEL_12;
    }
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - result) <= 5 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"-abort", 6u);
      result = *(_QWORD *)(a2 + 32);
    }
    else
    {
      *(_DWORD *)result = 1868718381;
      *(_WORD *)(result + 4) = 29810;
      result = *(_QWORD *)(a2 + 32) + 6LL;
      *(_QWORD *)(a2 + 32) = result;
    }
  }
  else if ( v14 <= 3 )
  {
    sub_CB6200(a2, (unsigned __int8 *)"trap", 4u);
    result = *(_QWORD *)(a2 + 32);
  }
  else
  {
    *v12 = 1885434484;
    result = *(_QWORD *)(a2 + 32) + 4LL;
    *(_QWORD *)(a2 + 32) = result;
  }
LABEL_12:
  if ( a1[3] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - result) <= 5 )
    {
      sub_CB6200(a2, ";merge", 6u);
      result = *(_QWORD *)(a2 + 32);
    }
    else
    {
      *(_DWORD *)result = 1919249723;
      *(_WORD *)(result + 4) = 25959;
      result = *(_QWORD *)(a2 + 32) + 6LL;
      *(_QWORD *)(a2 + 32) = result;
    }
  }
  if ( a1[5] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - result) <= 6 )
    {
      v17 = sub_CB6200(a2, ";guard=", 7u);
      sub_CB59F0(v17, (char)a1[4]);
    }
    else
    {
      *(_DWORD *)result = 1635084091;
      *(_WORD *)(result + 4) = 25714;
      *(_BYTE *)(result + 6) = 61;
      *(_QWORD *)(a2 + 32) += 7LL;
      sub_CB59F0(a2, (char)a1[4]);
    }
    result = *(_QWORD *)(a2 + 32);
  }
  if ( *(_QWORD *)(a2 + 24) == result )
    return sub_CB6200(a2, (unsigned __int8 *)">", 1u);
  *(_BYTE *)result = 62;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
