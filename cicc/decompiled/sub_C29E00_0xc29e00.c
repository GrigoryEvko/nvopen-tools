// Function: sub_C29E00
// Address: 0xc29e00
//
__int64 __fastcall sub_C29E00(__int64 a1)
{
  char v1; // al
  unsigned __int64 v2; // rsi
  __int64 result; // rax

  v1 = qword_4F83B28;
  v2 = *(_QWORD *)(a1 + 208);
  *(_BYTE *)(a1 + 184) = qword_4F83B28;
  unk_4F838D0 = v1;
  if ( v2 >= *(_QWORD *)(a1 + 216) )
  {
LABEL_6:
    sub_C1AFD0();
    return 0;
  }
  else
  {
    while ( 1 )
    {
      result = sub_C29DF0(a1, v2);
      if ( (_DWORD)result )
        break;
      v2 = *(_QWORD *)(a1 + 208);
      if ( v2 >= *(_QWORD *)(a1 + 216) )
        goto LABEL_6;
    }
  }
  return result;
}
