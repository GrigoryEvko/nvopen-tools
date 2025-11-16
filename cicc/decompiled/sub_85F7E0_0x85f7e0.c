// Function: sub_85F7E0
// Address: 0x85f7e0
//
__int64 __fastcall sub_85F7E0(__int64 a1, int a2)
{
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // r14
  _QWORD *i; // rbx
  __int64 v7; // rdi

  v3 = qword_4F04C68[0] + 776LL * a2;
  sub_85EE10(a1, v3, *(_DWORD *)(a1 + 56));
  result = (unsigned int)*(unsigned __int8 *)(v3 + 4) - 3;
  if ( (unsigned __int8)(*(_BYTE *)(v3 + 4) - 3) <= 1u )
  {
    result = *(_QWORD *)(v3 + 184);
    v5 = *(_QWORD *)(result + 32);
    if ( (*(_BYTE *)(v5 + 124) & 1) != 0 )
    {
      result = sub_735B70(*(_QWORD *)(result + 32));
      v5 = result;
    }
    while ( 1 )
    {
      for ( i = *(_QWORD **)(v3 + 536); i; i = (_QWORD *)*i )
      {
        result = i[2];
        v7 = *(_QWORD *)(result + 24);
        if ( (*(_BYTE *)(v7 + 124) & 1) != 0 )
        {
          result = sub_735B70(v7);
          if ( v5 == result )
          {
LABEL_11:
            result = sub_85EE10(a1, v3, dword_4F066AC);
            break;
          }
        }
        else if ( v5 == v7 )
        {
          goto LABEL_11;
        }
      }
      if ( !*(_BYTE *)(v3 + 4) )
        break;
      v3 -= 776;
    }
  }
  return result;
}
