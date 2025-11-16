// Function: sub_869ED0
// Address: 0x869ed0
//
_QWORD *__fastcall sub_869ED0(__int64 a1, _QWORD *a2, int a3, __int64 a4)
{
  __int64 v4; // rdx
  _QWORD *result; // rax

  if ( a3 == -1 )
  {
    if ( !a4 )
      BUG();
    result = *(_QWORD **)(a4 + 8);
    v4 = 0;
    if ( !result )
    {
      result = (_QWORD *)qword_4F07288;
      *(_QWORD *)(qword_4F07288 + 256) = a1;
      *(_QWORD *)(a1 + 8) = 0;
      goto LABEL_6;
    }
    goto LABEL_4;
  }
  v4 = qword_4F04C68[0] + 776LL * a3;
  if ( a4 )
  {
    result = *(_QWORD **)(a4 + 8);
    if ( result )
    {
LABEL_4:
      *result = a1;
      goto LABEL_5;
    }
    if ( !v4 )
    {
      MEMORY[0x100] = a1;
      BUG();
    }
  }
  else
  {
    result = *(_QWORD **)(v4 + 336);
    if ( result )
      goto LABEL_4;
  }
  *(_QWORD *)(v4 + 328) = a1;
  result = 0;
LABEL_5:
  *(_QWORD *)(a1 + 8) = result;
  if ( a4 )
  {
LABEL_6:
    *(_QWORD *)(a4 + 8) = a2;
    *a2 = a4;
    return result;
  }
  *(_QWORD *)(v4 + 336) = a2;
  *a2 = 0;
  return result;
}
