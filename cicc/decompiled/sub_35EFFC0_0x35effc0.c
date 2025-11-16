// Function: sub_35EFFC0
// Address: 0x35effc0
//
unsigned __int64 __fastcall sub_35EFFC0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  char v4; // al
  __int64 v5; // rdx
  unsigned __int64 result; // rax

  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  v5 = *(_QWORD *)(a4 + 32);
  switch ( v4 )
  {
    case 0:
      result = *(_QWORD *)(a4 + 24) - v5;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, (unsigned __int8 *)".add", 4u);
      }
      else
      {
        *(_DWORD *)v5 = 1684300078;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 1:
      result = *(_QWORD *)(a4 + 24) - v5;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, (unsigned __int8 *)".min", 4u);
      }
      else
      {
        *(_DWORD *)v5 = 1852402990;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 2:
      result = *(_QWORD *)(a4 + 24) - v5;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, (unsigned __int8 *)".max", 4u);
      }
      else
      {
        *(_DWORD *)v5 = 2019650862;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 3:
      result = *(_QWORD *)(a4 + 24) - v5;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, (unsigned __int8 *)".inc", 4u);
      }
      else
      {
        *(_DWORD *)v5 = 1668180270;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 4:
      result = *(_QWORD *)(a4 + 24) - v5;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, ".dec", 4u);
      }
      else
      {
        *(_DWORD *)v5 = 1667589166;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 5:
      result = *(_QWORD *)(a4 + 24) - v5;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, (unsigned __int8 *)".and", 4u);
      }
      else
      {
        *(_DWORD *)v5 = 1684955438;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 6:
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v5) <= 2 )
      {
        result = sub_CB6200(a4, (unsigned __int8 *)".or", 3u);
      }
      else
      {
        *(_BYTE *)(v5 + 2) = 114;
        *(_WORD *)v5 = 28462;
        *(_QWORD *)(a4 + 32) += 3LL;
        result = 28462;
      }
      break;
    case 7:
      result = *(_QWORD *)(a4 + 24) - v5;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, (unsigned __int8 *)".xor", 4u);
      }
      else
      {
        *(_DWORD *)v5 = 1919907886;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    default:
      BUG();
  }
  return result;
}
