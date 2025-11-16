// Function: sub_A58460
// Address: 0xa58460
//
_BYTE *__fastcall sub_A58460(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rcx
  __int64 *v5; // r15
  char v6; // r14
  __int64 v7; // r8
  _WORD *v8; // rdx
  _BYTE *result; // rax
  _BYTE *v10; // rax
  __int64 v11; // [rsp+0h] [rbp-40h]
  __int64 *v12; // [rsp+8h] [rbp-38h]

  if ( (*(_DWORD *)(a2 + 8) & 0x100) == 0 )
    return (_BYTE *)sub_904010(a3, "opaque");
  if ( ((*(_DWORD *)(a2 + 8) >> 8) & 2) != 0 )
  {
    v10 = *(_BYTE **)(a3 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(a3 + 24) )
    {
      sub_CB5D20(a3, 60);
    }
    else
    {
      *(_QWORD *)(a3 + 32) = v10 + 1;
      *v10 = 60;
    }
  }
  if ( !*(_DWORD *)(a2 + 12) )
  {
    result = (_BYTE *)sub_904010(a3, "{}");
    if ( (*(_BYTE *)(a2 + 9) & 2) == 0 )
      return result;
    goto LABEL_15;
  }
  sub_904010(a3, "{ ");
  v4 = *(__int64 **)(a2 + 16);
  v12 = &v4[*(unsigned int *)(a2 + 12)];
  if ( v12 != v4 )
  {
    v5 = *(__int64 **)(a2 + 16);
    v6 = 1;
    do
    {
      v7 = *v5;
      if ( v6 )
      {
        v6 = 0;
      }
      else
      {
        v8 = *(_WORD **)(a3 + 32);
        if ( *(_QWORD *)(a3 + 24) - (_QWORD)v8 > 1u )
        {
          *v8 = 8236;
          *(_QWORD *)(a3 + 32) += 2LL;
        }
        else
        {
          v11 = *v5;
          sub_CB6200(a3, ", ", 2);
          v7 = v11;
        }
      }
      ++v5;
      sub_A57EC0(a1, v7, a3);
    }
    while ( v12 != v5 );
  }
  result = (_BYTE *)sub_904010(a3, " }");
  if ( (*(_BYTE *)(a2 + 9) & 2) != 0 )
  {
LABEL_15:
    result = *(_BYTE **)(a3 + 32);
    if ( (unsigned __int64)result >= *(_QWORD *)(a3 + 24) )
    {
      return (_BYTE *)sub_CB5D20(a3, 62);
    }
    else
    {
      *(_QWORD *)(a3 + 32) = result + 1;
      *result = 62;
    }
  }
  return result;
}
