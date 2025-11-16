// Function: sub_A6AC90
// Address: 0xa6ac90
//
_BYTE *__fastcall sub_A6AC90(__int64 *a1, __int64 *a2, const char *a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rbx
  char v8; // r13
  _WORD *v9; // rdx
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rdi
  _WORD *v13; // rdx
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rdi
  _BYTE *result; // rax

  v4 = sub_904010(*a1, a3);
  v5 = *(_QWORD *)(v4 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 24) - v5) <= 2 )
  {
    sub_CB6200(v4, ": (", 3);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 40;
    *(_WORD *)v5 = 8250;
    *(_QWORD *)(v4 + 32) += 3LL;
  }
  v6 = a2[1];
  v7 = *a2;
  v8 = 1;
  if ( *a2 != v6 )
  {
    while ( 1 )
    {
      v10 = *a1;
      if ( v8 )
      {
        v11 = *(_BYTE **)(v10 + 32);
        v8 = 0;
        if ( *(_BYTE **)(v10 + 24) == v11 )
          goto LABEL_17;
      }
      else
      {
        v9 = *(_WORD **)(v10 + 32);
        if ( *(_QWORD *)(v10 + 24) - (_QWORD)v9 <= 1u )
        {
          sub_CB6200(v10, ", ", 2);
        }
        else
        {
          *v9 = 8236;
          *(_QWORD *)(v10 + 32) += 2LL;
        }
        v10 = *a1;
        v11 = *(_BYTE **)(*a1 + 32);
        if ( *(_BYTE **)(*a1 + 24) == v11 )
        {
LABEL_17:
          sub_CB6200(v10, "(", 1);
          goto LABEL_9;
        }
      }
      *v11 = 40;
      ++*(_QWORD *)(v10 + 32);
LABEL_9:
      sub_A6A880(a1, *(_QWORD *)v7, *(_QWORD *)(v7 + 8));
      if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(v7 + 16) )
      {
        v12 = *a1;
        v13 = *(_WORD **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v13 <= 1u )
        {
          sub_CB6200(v12, ", ", 2);
        }
        else
        {
          *v13 = 8236;
          *(_QWORD *)(v12 + 32) += 2LL;
        }
        sub_A50F50(a1, (__int64 **)(v7 + 16));
      }
      v14 = *a1;
      v15 = *(_BYTE **)(*a1 + 32);
      if ( *(_BYTE **)(*a1 + 24) == v15 )
      {
        v7 += 40;
        sub_CB6200(v14, ")", 1);
        if ( v6 == v7 )
          break;
      }
      else
      {
        v7 += 40;
        *v15 = 41;
        ++*(_QWORD *)(v14 + 32);
        if ( v6 == v7 )
          break;
      }
    }
  }
  v16 = *a1;
  result = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == result )
    return (_BYTE *)sub_CB6200(v16, ")", 1);
  *result = 41;
  ++*(_QWORD *)(v16 + 32);
  return result;
}
