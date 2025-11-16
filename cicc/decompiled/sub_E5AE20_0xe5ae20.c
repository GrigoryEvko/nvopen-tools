// Function: sub_E5AE20
// Address: 0xe5ae20
//
_BYTE *__fastcall sub_E5AE20(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *result; // rax
  char v7; // dl
  const char *v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // r13
  size_t v11; // rax
  void *v12; // rdi
  size_t v13; // r15
  unsigned __int64 v14; // r13
  bool v15; // zf
  __int64 v16; // rdi
  __int64 v17; // r15
  unsigned __int8 *v18; // rsi
  size_t v19; // rdx
  void *v20; // rdi
  __int64 v21; // rax
  char *src; // [rsp+8h] [rbp-48h]
  __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  result = (_BYTE *)sub_E81180(a2, v23);
  v7 = (char)result;
  if ( (_BYTE)result && !v23[0] )
    return result;
  result = *(_BYTE **)(a1 + 312);
  v8 = (const char *)*((_QWORD *)result + 24);
  if ( !v8 )
    return (_BYTE *)nullsub_361(a1, a2, a3, 0);
  if ( !result[21] )
  {
    sub_904010(*(_QWORD *)(a1 + 304), v8);
    sub_E7FAD0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312), 0);
    if ( a3 )
    {
      v21 = sub_A51310(*(_QWORD *)(a1 + 304), 0x2Cu);
      sub_CB59F0(v21, (int)a3);
    }
    return sub_E4D880(a1);
  }
  if ( !a3 )
  {
    sub_904010(*(_QWORD *)(a1 + 304), v8);
    sub_E7FAD0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312), 0);
    return sub_E4D880(a1);
  }
  if ( !v7 )
    sub_C64ED0("Cannot emit non-absolute expression lengths of fill.", 1u);
  if ( v23[0] > 0 )
  {
    v9 = 0;
    while ( 1 )
    {
      v10 = *(_QWORD *)(a1 + 304);
      if ( *((_QWORD *)result + 28) )
      {
        src = (char *)*((_QWORD *)result + 28);
        v11 = strlen(src);
        v12 = *(void **)(v10 + 32);
        v13 = v11;
        if ( v11 > *(_QWORD *)(v10 + 24) - (_QWORD)v12 )
        {
          v10 = sub_CB6200(v10, (unsigned __int8 *)src, v11);
        }
        else if ( v11 )
        {
          memcpy(v12, src, v11);
          *(_QWORD *)(v10 + 32) += v13;
        }
      }
      sub_CB59F0(v10, (int)a3);
      v14 = *(_QWORD *)(a1 + 344);
      if ( v14 )
      {
        v17 = *(_QWORD *)(a1 + 304);
        v18 = *(unsigned __int8 **)(a1 + 336);
        v19 = *(_QWORD *)(a1 + 344);
        v20 = *(void **)(v17 + 32);
        if ( v14 > *(_QWORD *)(v17 + 24) - (_QWORD)v20 )
        {
          sub_CB6200(*(_QWORD *)(a1 + 304), v18, v19);
        }
        else
        {
          memcpy(v20, v18, v19);
          *(_QWORD *)(v17 + 32) += v14;
        }
      }
      v15 = *(_BYTE *)(a1 + 745) == 0;
      *(_QWORD *)(a1 + 344) = 0;
      if ( !v15 )
        break;
      v16 = *(_QWORD *)(a1 + 304);
      result = *(_BYTE **)(v16 + 32);
      if ( (unsigned __int64)result >= *(_QWORD *)(v16 + 24) )
      {
        result = (_BYTE *)sub_CB5D20(v16, 10);
LABEL_10:
        if ( v23[0] <= ++v9 )
          return result;
        goto LABEL_11;
      }
      ++v9;
      *(_QWORD *)(v16 + 32) = result + 1;
      *result = 10;
      if ( v23[0] <= v9 )
        return result;
LABEL_11:
      result = *(_BYTE **)(a1 + 312);
    }
    result = (_BYTE *)sub_E4D630((__int64 *)a1);
    goto LABEL_10;
  }
  return result;
}
