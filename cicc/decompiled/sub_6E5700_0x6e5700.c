// Function: sub_6E5700
// Address: 0x6e5700
//
__int64 __fastcall sub_6E5700(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rdx
  __int64 result; // rax
  __int64 v3; // r12
  char v4; // dl
  unsigned __int64 v5; // rax

  v1 = a1[2];
  result = *(unsigned __int8 *)(v1 + 80);
  switch ( (_BYTE)result )
  {
    case 7:
      v3 = *(_QWORD *)(v1 + 88);
      if ( dword_4F077C4 != 2 && *(_BYTE *)(v3 + 136) == 5 )
      {
        if ( !unk_4D0436C && byte_4F07472[0] == 8 )
        {
          if ( (unsigned int)sub_6E5430() )
            sub_6851C0(0x8Au, (_DWORD *)a1 + 8);
          v5 = *a1;
          *((_BYTE *)a1 + 8) = 0;
          *a1 = v5 & 0xFFFFFFFFFFFECF87LL | 0x40;
        }
        else if ( sub_6E53E0(5, 0x8Au, (_DWORD *)a1 + 8) )
        {
          sub_684B30(0x8Au, (_DWORD *)a1 + 8);
        }
      }
      return sub_72A420(v3);
    case 9:
      return sub_72A420(*(_QWORD *)(v1 + 88));
    case 8:
      result = *(_QWORD *)(v1 + 96);
      if ( result )
      {
        while ( 1 )
        {
          v4 = *(_BYTE *)(result + 80);
          if ( v4 == 7 )
            break;
          if ( v4 == 8 )
          {
            result = *(_QWORD *)(result + 96);
            if ( result )
              continue;
          }
          return result;
        }
        return sub_72A420(*(_QWORD *)(result + 88));
      }
      break;
  }
  return result;
}
