// Function: sub_F34950
// Address: 0xf34950
//
__int64 __fastcall sub_F34950(unsigned __int8 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  unsigned __int8 *v5; // rdi

  result = *a1;
  switch ( (_BYTE)result )
  {
    case '"':
      if ( *((_QWORD *)a1 - 8) )
      {
        result = *((_QWORD *)a1 - 7);
        **((_QWORD **)a1 - 6) = result;
        if ( result )
          *(_QWORD *)(result + 16) = *((_QWORD *)a1 - 6);
      }
      *((_QWORD *)a1 - 8) = a2;
      if ( a2 )
      {
        result = *(_QWORD *)(a2 + 16);
        *((_QWORD *)a1 - 7) = result;
        if ( result )
          *(_QWORD *)(result + 16) = a1 - 56;
        *((_QWORD *)a1 - 6) = a2 + 16;
        *(_QWORD *)(a2 + 16) = a1 - 64;
      }
      break;
    case '\'':
      result = *((_QWORD *)a1 - 1);
      if ( *(_QWORD *)(result + 32) )
      {
        v3 = *(_QWORD *)(result + 40);
        **(_QWORD **)(result + 48) = v3;
        if ( v3 )
          *(_QWORD *)(v3 + 16) = *(_QWORD *)(result + 48);
      }
      *(_QWORD *)(result + 32) = a2;
      if ( a2 )
      {
        v4 = *(_QWORD *)(a2 + 16);
        *(_QWORD *)(result + 40) = v4;
        if ( v4 )
          *(_QWORD *)(v4 + 16) = result + 40;
        *(_QWORD *)(result + 48) = a2 + 16;
        result += 32;
        *(_QWORD *)(a2 + 16) = result;
      }
      break;
    case '%':
      result = 32 * (1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF));
      v5 = &a1[result];
      if ( *(_QWORD *)v5 )
      {
        result = *((_QWORD *)v5 + 1);
        **((_QWORD **)v5 + 2) = result;
        if ( result )
          *(_QWORD *)(result + 16) = *((_QWORD *)v5 + 2);
      }
      *(_QWORD *)v5 = a2;
      if ( a2 )
      {
        result = *(_QWORD *)(a2 + 16);
        *((_QWORD *)v5 + 1) = result;
        if ( result )
          *(_QWORD *)(result + 16) = v5 + 8;
        *((_QWORD *)v5 + 2) = a2 + 16;
        *(_QWORD *)(a2 + 16) = v5;
      }
      break;
    default:
      BUG();
  }
  return result;
}
