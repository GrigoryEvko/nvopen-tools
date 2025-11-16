// Function: sub_BBB600
// Address: 0xbbb600
//
_BYTE *__fastcall sub_BBB600(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // rdx
  _BYTE *v7; // rax
  _BYTE *result; // rax

  v6 = *(_QWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 > 7u )
  {
    *v6 = 0x6E6F6974636E7566LL;
    v7 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 8LL);
    *(_QWORD *)(a2 + 32) = v7;
    if ( !a1[8] )
      goto LABEL_3;
LABEL_8:
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 0xAu )
    {
      sub_CB6200(a2, "<eager-inv>", 11);
      v7 = *(_BYTE **)(a2 + 32);
    }
    else
    {
      qmemcpy(v7, "<eager-inv>", 11);
      v7 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 11LL);
      *(_QWORD *)(a2 + 32) = v7;
    }
    goto LABEL_3;
  }
  sub_CB6200(a2, "function", 8);
  v7 = *(_BYTE **)(a2 + 32);
  if ( a1[8] )
    goto LABEL_8;
LABEL_3:
  if ( *(_QWORD *)(a2 + 24) <= (unsigned __int64)v7 )
  {
    sub_CB5D20(a2, 40);
  }
  else
  {
    *(_QWORD *)(a2 + 32) = v7 + 1;
    *v7 = 40;
  }
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)a1 + 24LL))(*(_QWORD *)a1, a2, a3, a4);
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 41);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 41;
  return result;
}
