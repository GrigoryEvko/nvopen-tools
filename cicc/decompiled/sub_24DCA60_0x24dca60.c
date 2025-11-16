// Function: sub_24DCA60
// Address: 0x24dca60
//
_BYTE *__fastcall sub_24DCA60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  _BYTE *v7; // rax
  __int64 v8; // rax
  unsigned int v9; // ebx
  __int64 v10; // rdx
  _BYTE *v11; // rax
  _BYTE *result; // rax
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  v6 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v6) <= 8 )
  {
    sub_CB6200(a2, "coro-cond", 9u);
    v7 = *(_BYTE **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) > (unsigned __int64)v7 )
      goto LABEL_3;
  }
  else
  {
    *(_BYTE *)(v6 + 8) = 100;
    *(_QWORD *)v6 = 0x6E6F632D6F726F63LL;
    v7 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 9LL);
    *(_QWORD *)(a2 + 32) = v7;
    if ( *(_QWORD *)(a2 + 24) > (unsigned __int64)v7 )
    {
LABEL_3:
      *(_QWORD *)(a2 + 32) = v7 + 1;
      *v7 = 40;
      goto LABEL_4;
    }
  }
  sub_CB5D20(a2, 40);
LABEL_4:
  v8 = *a1;
  v13 = (a1[1] - *a1) >> 3;
  if ( v13 )
  {
    v9 = 0;
    while ( 1 )
    {
      v10 = v9++;
      (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v8 + 8 * v10) + 24LL))(
        *(_QWORD *)(v8 + 8 * v10),
        a2,
        a3,
        a4);
      if ( v13 <= v9 )
      {
        if ( v13 == v9 )
          break;
      }
      else
      {
        v11 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v11 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 44);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v11 + 1;
          *v11 = 44;
        }
      }
      v8 = *a1;
    }
  }
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 41);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 41;
  return result;
}
