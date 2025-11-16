// Function: sub_735890
// Address: 0x735890
//
__int64 __fastcall sub_735890(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax
  __int64 v4; // rax
  int v5; // [rsp+4h] [rbp-1Ch]
  int v6; // [rsp+4h] [rbp-1Ch]

  if ( a2 )
  {
    result = *(_QWORD *)(a2 + 40);
    if ( result )
    {
LABEL_3:
      *(_QWORD *)(result + 32) = a1;
      *(_QWORD *)(result + 40) = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(result + 56) = *(_QWORD *)(a1 + 48);
      *(_QWORD *)(a1 + 48) = result;
      goto LABEL_4;
    }
    result = *(unsigned __int8 *)(a2 + 48);
    if ( (_BYTE)result == 3 )
    {
      v4 = *(_QWORD *)(a2 + 56);
      if ( *(_BYTE *)(v4 + 24) == 10 )
      {
        result = *(_QWORD *)(v4 + 64);
        if ( result )
          goto LABEL_3;
      }
      goto LABEL_11;
    }
  }
  else
  {
    result = MEMORY[0x30];
  }
  if ( (_BYTE)result == 6 )
  {
LABEL_16:
    v6 = a3;
    result = sub_735980(a1, *(_QWORD *)(a2 + 56));
    a3 = v6;
    goto LABEL_4;
  }
  if ( (_BYTE)result == 8 )
  {
    if ( (*(_BYTE *)(a2 + 72) & 1) == 0 )
      goto LABEL_4;
    goto LABEL_16;
  }
  result = (unsigned int)(result - 3);
  if ( (unsigned __int8)result <= 1u )
  {
LABEL_11:
    v5 = a3;
    result = sub_735A00(a1, *(_QWORD *)(a2 + 56));
    a3 = v5;
  }
LABEL_4:
  if ( !a3 )
  {
    if ( *(_QWORD *)(a2 + 16) )
    {
      result = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a2 + 32) = result;
      *(_QWORD *)(a1 + 24) = a2;
      *(_QWORD *)(a2 + 24) = a1;
    }
  }
  return result;
}
