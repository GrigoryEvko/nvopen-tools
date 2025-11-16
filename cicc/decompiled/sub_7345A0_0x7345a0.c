// Function: sub_7345A0
// Address: 0x7345a0
//
__int64 __fastcall sub_7345A0(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // r11
  __int64 v3; // rbx
  __int64 v4; // r8
  __int64 v5; // rdi
  _QWORD *v6; // r9
  __int64 v7; // r10
  __int64 v8; // r10
  __int64 v9; // rdi

  result = sub_728280(a1);
  if ( result )
  {
    v3 = *(_QWORD *)(unk_4F066A0 + 8LL * *(int *)(v2 + 24));
    result = *(_QWORD *)(v3 + 8);
    v4 = *(_QWORD *)(result + 232);
    if ( v4 )
    {
      while ( 1 )
      {
        result = *(unsigned __int8 *)(v4 + 16);
        if ( (_BYTE)result == 7 )
        {
          v5 = *(_QWORD *)(v4 + 24);
          if ( (*(_BYTE *)(v5 + 89) & 1) == 0 )
            goto LABEL_5;
          result = sub_727F30(v5, v2);
          if ( !result )
            goto LABEL_5;
          if ( !v6 )
          {
LABEL_19:
            *(_QWORD *)(unk_4F07288 + 232LL) = *(_QWORD *)v4;
            goto LABEL_11;
          }
        }
        else
        {
          if ( (_BYTE)result != 6 )
            goto LABEL_5;
          v9 = *(_QWORD *)(v4 + 24);
          if ( (*(_BYTE *)(v9 + 89) & 1) == 0 )
            goto LABEL_5;
          result = sub_728070(v9, v2);
          if ( !result )
            goto LABEL_5;
          if ( !v6 )
            goto LABEL_19;
        }
        *v6 = *(_QWORD *)v4;
LABEL_11:
        result = *(_QWORD *)v4;
        v8 = v7 - 1;
        if ( *(_QWORD *)v4 )
        {
          if ( !v8 )
            return result;
          v4 = *(_QWORD *)v4;
        }
        else
        {
          *(_QWORD *)(v3 + 120) = v6;
          if ( !v8 )
            return result;
LABEL_5:
          v4 = *(_QWORD *)v4;
          if ( !v4 )
            return result;
        }
      }
    }
  }
  return result;
}
