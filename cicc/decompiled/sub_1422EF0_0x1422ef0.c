// Function: sub_1422EF0
// Address: 0x1422ef0
//
__int64 __fastcall sub_1422EF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  void *v5; // rdx
  __int64 v6; // rsi
  __int64 result; // rax
  int v8; // edx
  void *v9; // rdx
  _WORD *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int8 v13; // bl
  _BYTE *v14; // rax
  void *v15; // rdx

  v2 = a2;
  v3 = *(_QWORD *)(a1 - 24);
  v4 = sub_16E7A90(a2, *(unsigned int *)(a1 + 84));
  v5 = *(void **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0xCu )
  {
    sub_16E7EE0(v4, " = MemoryDef(", 13);
  }
  else
  {
    qmemcpy(v5, " = MemoryDef(", 13);
    *(_QWORD *)(v4 + 24) += 13LL;
  }
  if ( v3
    && (*(_BYTE *)(v3 + 16) != 22 ? (v6 = *(unsigned int *)(v3 + 72)) : (v6 = *(unsigned int *)(v3 + 84)), (_DWORD)v6) )
  {
    sub_16E7A90(v2, v6);
    result = *(_QWORD *)(v2 + 24);
  }
  else
  {
    v9 = *(void **)(v2 + 24);
    if ( *(_QWORD *)(v2 + 16) - (_QWORD)v9 > 0xAu )
    {
      qmemcpy(v9, "liveOnEntry", 11);
      result = *(_QWORD *)(v2 + 24) + 11LL;
      *(_QWORD *)(v2 + 24) = result;
      if ( *(_QWORD *)(v2 + 16) != result )
        goto LABEL_9;
LABEL_17:
      result = sub_16E7EE0(v2, ")", 1);
      goto LABEL_10;
    }
    sub_16E7EE0(v2, "liveOnEntry", 11);
    result = *(_QWORD *)(v2 + 24);
  }
  if ( *(_QWORD *)(v2 + 16) == result )
    goto LABEL_17;
LABEL_9:
  *(_BYTE *)result = 41;
  ++*(_QWORD *)(v2 + 24);
LABEL_10:
  if ( *(_QWORD *)(a1 + 112) )
  {
    result = *(_QWORD *)(a1 - 24);
    if ( result )
    {
      v8 = *(_DWORD *)(a1 + 88);
      if ( *(_BYTE *)(result + 16) == 22 )
      {
        result = *(unsigned int *)(result + 84);
        if ( v8 != (_DWORD)result )
          return result;
      }
      else
      {
        result = *(unsigned int *)(result + 72);
        if ( v8 != (_DWORD)result )
          return result;
      }
      v10 = *(_WORD **)(v2 + 24);
      if ( *(_QWORD *)(v2 + 16) - (_QWORD)v10 <= 1u )
      {
        sub_16E7EE0(v2, "->", 2);
      }
      else
      {
        *v10 = 15917;
        *(_QWORD *)(v2 + 24) += 2LL;
      }
      v11 = *(_QWORD *)(a1 + 112);
      if ( v11
        && (*(_BYTE *)(v11 + 16) != 22 ? (v12 = *(unsigned int *)(v11 + 72)) : (v12 = *(unsigned int *)(v11 + 84)),
            (_DWORD)v12) )
      {
        result = sub_16E7A90(v2, v12);
      }
      else
      {
        v15 = *(void **)(v2 + 24);
        if ( *(_QWORD *)(v2 + 16) - (_QWORD)v15 <= 0xAu )
        {
          result = sub_16E7EE0(v2, "liveOnEntry", 11);
        }
        else
        {
          qmemcpy(v15, "liveOnEntry", 11);
          result = 29300;
          *(_QWORD *)(v2 + 24) += 11LL;
        }
      }
      if ( *(_BYTE *)(a1 + 81) )
      {
        v13 = *(_BYTE *)(a1 + 80);
        v14 = *(_BYTE **)(v2 + 24);
        if ( *(_BYTE **)(v2 + 16) == v14 )
        {
          v2 = sub_16E7EE0(v2, " ", 1);
        }
        else
        {
          *v14 = 32;
          ++*(_QWORD *)(v2 + 24);
        }
        return sub_134CED0(v2, v13);
      }
    }
  }
  return result;
}
