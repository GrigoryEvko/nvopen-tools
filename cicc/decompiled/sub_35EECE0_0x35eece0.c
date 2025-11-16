// Function: sub_35EECE0
// Address: 0x35eece0
//
size_t __fastcall sub_35EECE0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  size_t result; // rax
  _DWORD *v9; // rdx
  __int64 v10; // rdx
  size_t v11; // rdx
  char *v12; // rsi

  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( !a5 )
    goto LABEL_67;
  result = strlen((const char *)a5);
  if ( result == 3 )
  {
    if ( *(_WORD *)a5 == 29798 && *(_BYTE *)(a5 + 2) == 122 )
    {
      if ( (v5 & 0x100) != 0 )
      {
        v9 = *(_DWORD **)(a4 + 32);
        result = *(_QWORD *)(a4 + 24) - (_QWORD)v9;
        if ( result <= 3 )
        {
          v11 = 4;
          v12 = ".ftz";
          return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
        }
        else
        {
          *v9 = 2054448686;
          *(_QWORD *)(a4 + 32) += 4LL;
        }
      }
      return result;
    }
LABEL_67:
    BUG();
  }
  if ( result != 4 || *(_DWORD *)a5 != 1702060386 )
    goto LABEL_67;
  result = (unsigned __int8)v5;
  v10 = *(_QWORD *)(a4 + 32);
  switch ( (char)v5 )
  {
    case 0:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 2 )
      {
        v11 = 3;
        v12 = ".eq";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 113;
      *(_WORD *)v10 = 25902;
      *(_QWORD *)(a4 + 32) += 3LL;
      break;
    case 1:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 2 )
      {
        v11 = 3;
        v12 = ".ne";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 101;
      *(_WORD *)v10 = 28206;
      *(_QWORD *)(a4 + 32) += 3LL;
      break;
    case 2:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 2 )
      {
        v11 = 3;
        v12 = ".lt";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 116;
      *(_WORD *)v10 = 27694;
      *(_QWORD *)(a4 + 32) += 3LL;
      break;
    case 3:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 2 )
      {
        v11 = 3;
        v12 = ".le";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 101;
      *(_WORD *)v10 = 27694;
      *(_QWORD *)(a4 + 32) += 3LL;
      break;
    case 4:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 2 )
      {
        v11 = 3;
        v12 = ".gt";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 116;
      *(_WORD *)v10 = 26414;
      *(_QWORD *)(a4 + 32) += 3LL;
      break;
    case 5:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 2 )
      {
        v11 = 3;
        v12 = ".ge";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 101;
      *(_WORD *)v10 = 26414;
      *(_QWORD *)(a4 + 32) += 3LL;
      break;
    case 6:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 2 )
      {
        v11 = 3;
        v12 = ".lo";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 111;
      *(_WORD *)v10 = 27694;
      *(_QWORD *)(a4 + 32) += 3LL;
      break;
    case 7:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 2 )
      {
        v11 = 3;
        v12 = asc_432C6C4;
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 115;
      *(_WORD *)v10 = 27694;
      *(_QWORD *)(a4 + 32) += 3LL;
      break;
    case 8:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 2 )
      {
        v11 = 3;
        v12 = ".hi";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 105;
      *(_WORD *)v10 = 26670;
      *(_QWORD *)(a4 + 32) += 3LL;
      break;
    case 9:
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v10) <= 2 )
      {
        v11 = 3;
        v12 = ".hs";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_BYTE *)(v10 + 2) = 115;
      *(_WORD *)v10 = 26670;
      *(_QWORD *)(a4 + 32) += 3LL;
      result = 26670;
      break;
    case 10:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 3 )
      {
        v11 = 4;
        v12 = ".equ";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_DWORD *)v10 = 1970365742;
      *(_QWORD *)(a4 + 32) += 4LL;
      break;
    case 11:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 3 )
      {
        v11 = 4;
        v12 = ".neu";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_DWORD *)v10 = 1969581614;
      *(_QWORD *)(a4 + 32) += 4LL;
      break;
    case 12:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 3 )
      {
        v11 = 4;
        v12 = ".ltu";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_DWORD *)v10 = 1970564142;
      *(_QWORD *)(a4 + 32) += 4LL;
      break;
    case 13:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 3 )
      {
        v11 = 4;
        v12 = ".leu";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_DWORD *)v10 = 1969581102;
      *(_QWORD *)(a4 + 32) += 4LL;
      break;
    case 14:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 3 )
      {
        v11 = 4;
        v12 = ".gtu";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_DWORD *)v10 = 1970562862;
      *(_QWORD *)(a4 + 32) += 4LL;
      break;
    case 15:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 3 )
      {
        v11 = 4;
        v12 = ".geu";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_DWORD *)v10 = 1969579822;
      *(_QWORD *)(a4 + 32) += 4LL;
      break;
    case 16:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 3 )
      {
        v11 = 4;
        v12 = ".num";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_DWORD *)v10 = 1836412462;
      *(_QWORD *)(a4 + 32) += 4LL;
      break;
    case 17:
      result = *(_QWORD *)(a4 + 24) - v10;
      if ( result <= 3 )
      {
        v11 = 4;
        v12 = ".nan";
        return sub_CB6200(a4, (unsigned __int8 *)v12, v11);
      }
      *(_DWORD *)v10 = 1851878958;
      *(_QWORD *)(a4 + 32) += 4LL;
      break;
    default:
      return result;
  }
  return result;
}
