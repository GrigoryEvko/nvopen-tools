// Function: sub_F08290
// Address: 0xf08290
//
__int64 __fastcall sub_F08290(__int64 a1)
{
  __int64 v2; // rbx
  unsigned __int8 *v3; // rdi
  int v4; // eax
  __int64 result; // rax
  int v6; // eax
  unsigned __int8 *v7; // rdx

  v2 = *(_QWORD *)(a1 + 16);
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    v3 = *(unsigned __int8 **)(v2 + 24);
    v4 = *v3;
    if ( (unsigned __int8)v4 > 0x1Cu )
      break;
    if ( (_BYTE)v4 != 5 )
      goto LABEL_4;
    v6 = *((unsigned __int16 *)v3 + 1);
LABEL_8:
    if ( v6 == 49 )
    {
      v7 = (v3[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v3 - 1) : &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
      if ( a1 == *(_QWORD *)v7 )
      {
        result = sub_F08290();
        if ( (_BYTE)result )
          return result;
      }
    }
LABEL_4:
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 0;
  }
  if ( (_BYTE)v4 != 92 )
  {
    v6 = v4 - 29;
    goto LABEL_8;
  }
  return 1;
}
