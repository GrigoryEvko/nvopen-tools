// Function: sub_1698E10
// Address: 0x1698e10
//
char __fastcall sub_1698E10(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  char result; // al
  __int64 v5; // rax
  unsigned int v6; // [rsp-Ch] [rbp-Ch]

  switch ( a2 )
  {
    case 0u:
      if ( (_DWORD)a3 == 3 || (result = 0, (_DWORD)a3 != 2) )
      {
        result = nullsub_2033(a1, a2, a3, a4, a1);
      }
      else
      {
        v6 = a4;
        if ( (*(_BYTE *)(a1 + 18) & 7) != 3 )
        {
          v5 = sub_16984A0(a1);
          result = (unsigned int)sub_16A70B0(v5, v6) != 0;
        }
      }
      break;
    case 1u:
      result = ((*(_BYTE *)(a1 + 18) >> 3) ^ 1) & 1;
      break;
    case 2u:
      result = (*(_BYTE *)(a1 + 18) & 8) != 0;
      break;
    case 3u:
      result = 0;
      break;
    case 4u:
      result = (unsigned int)(a3 - 2) <= 1;
      break;
  }
  return result;
}
