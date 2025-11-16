// Function: sub_88D9B0
// Address: 0x88d9b0
//
__int64 __fastcall sub_88D9B0(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  int v7; // ebx
  char v8; // r14
  unsigned int v9; // r15d
  __int64 v10; // rdx
  __int64 result; // rax

  v6 = (int)a2;
  v7 = a1;
  v8 = (a1 ^ 1) & 1;
  v9 = (_DWORD)a2 == 0 ? 1014 : 2868;
  while ( 1 )
  {
    v10 = word_4F06418[0];
    switch ( word_4F06418[0] )
    {
      case 0x51u:
      case 0x58u:
      case 0x64u:
      case 0x67u:
      case 0x6Bu:
      case 0x9Au:
        if ( v7 )
          goto LABEL_8;
        goto LABEL_5;
      case 0x57u:
      case 0x65u:
      case 0x68u:
      case 0x97u:
        result = word_4F06418[0] == 87;
        if ( (_DWORD)result == v6 || v8 )
          return result;
LABEL_8:
        a2 = dword_4F07508;
        a1 = v9;
        sub_6851C0(v9, dword_4F07508);
LABEL_5:
        sub_7B8B50(a1, a2, v10, a4, a5, a6);
        break;
      default:
        return (unsigned int)word_4F06418[0] - 81;
    }
  }
}
