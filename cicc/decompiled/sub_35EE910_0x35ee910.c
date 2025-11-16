// Function: sub_35EE910
// Address: 0x35ee910
//
size_t __fastcall sub_35EE910(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  size_t result; // rax
  _DWORD *v9; // rdx
  _DWORD *v10; // rdx
  __int64 v11; // rdx
  _DWORD *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  _DWORD *v17; // rdx
  _DWORD *v18; // rdx
  _DWORD *v19; // rdx
  _DWORD *v20; // rdx
  size_t v21; // rdx
  char *v22; // rsi

  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( !a5 )
    goto LABEL_52;
  result = strlen((const char *)a5);
  if ( result != 3 )
  {
    if ( result == 4 )
    {
      if ( *(_DWORD *)a5 == 1970038130 )
      {
        if ( (v5 & 0x40) != 0 )
        {
          v11 = *(_QWORD *)(a4 + 32);
          result = *(_QWORD *)(a4 + 24) - v11;
          if ( result > 4 )
          {
            *(_DWORD *)v11 = 1818587694;
            *(_BYTE *)(v11 + 4) = 117;
            *(_QWORD *)(a4 + 32) += 5LL;
            return result;
          }
          v21 = 5;
          v22 = ".relu";
          return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
        }
        return result;
      }
      if ( *(_DWORD *)a5 == 1702060386 )
      {
        switch ( v5 & 0xF )
        {
          case 1LL:
            v19 = *(_DWORD **)(a4 + 32);
            result = *(_QWORD *)(a4 + 24) - (_QWORD)v19;
            if ( result <= 3 )
            {
              v21 = 4;
              v22 = ".rni";
              return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
            }
            *v19 = 1768845870;
            *(_QWORD *)(a4 + 32) += 4LL;
            break;
          case 2LL:
            v20 = *(_DWORD **)(a4 + 32);
            result = *(_QWORD *)(a4 + 24) - (_QWORD)v20;
            if ( result <= 3 )
            {
              v21 = 4;
              v22 = ".rzi";
              return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
            }
            *v20 = 1769632302;
            *(_QWORD *)(a4 + 32) += 4LL;
            break;
          case 3LL:
            v18 = *(_DWORD **)(a4 + 32);
            result = *(_QWORD *)(a4 + 24) - (_QWORD)v18;
            if ( result <= 3 )
            {
              v21 = 4;
              v22 = ".rmi";
              return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
            }
            *v18 = 1768780334;
            *(_QWORD *)(a4 + 32) += 4LL;
            break;
          case 4LL:
            v17 = *(_DWORD **)(a4 + 32);
            result = *(_QWORD *)(a4 + 24) - (_QWORD)v17;
            if ( result <= 3 )
            {
              v21 = 4;
              v22 = ".rpi";
              return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
            }
            *v17 = 1768976942;
            *(_QWORD *)(a4 + 32) += 4LL;
            break;
          case 5LL:
            v15 = *(_QWORD *)(a4 + 32);
            result = *(_QWORD *)(a4 + 24) - v15;
            if ( result <= 2 )
            {
              v21 = 3;
              v22 = ".rn";
              return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
            }
            *(_BYTE *)(v15 + 2) = 110;
            *(_WORD *)v15 = 29230;
            *(_QWORD *)(a4 + 32) += 3LL;
            break;
          case 6LL:
            v16 = *(_QWORD *)(a4 + 32);
            result = *(_QWORD *)(a4 + 24) - v16;
            if ( result <= 2 )
            {
              v21 = 3;
              v22 = ".rz";
              return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
            }
            *(_BYTE *)(v16 + 2) = 122;
            *(_WORD *)v16 = 29230;
            *(_QWORD *)(a4 + 32) += 3LL;
            break;
          case 7LL:
            v14 = *(_QWORD *)(a4 + 32);
            result = *(_QWORD *)(a4 + 24) - v14;
            if ( result <= 2 )
            {
              v21 = 3;
              v22 = ".rm";
              return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
            }
            *(_BYTE *)(v14 + 2) = 109;
            *(_WORD *)v14 = 29230;
            *(_QWORD *)(a4 + 32) += 3LL;
            break;
          case 8LL:
            v13 = *(_QWORD *)(a4 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v13) <= 2 )
            {
              v21 = 3;
              v22 = ".rp";
              return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
            }
            *(_BYTE *)(v13 + 2) = 112;
            *(_WORD *)v13 = 29230;
            *(_QWORD *)(a4 + 32) += 3LL;
            result = 29230;
            break;
          case 9LL:
            v12 = *(_DWORD **)(a4 + 32);
            result = *(_QWORD *)(a4 + 24) - (_QWORD)v12;
            if ( result <= 3 )
            {
              v21 = 4;
              v22 = ".rna";
              return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
            }
            *v12 = 1634628142;
            *(_QWORD *)(a4 + 32) += 4LL;
            break;
          default:
            return result;
        }
        return result;
      }
    }
    goto LABEL_52;
  }
  if ( *(_WORD *)a5 == 29798 && *(_BYTE *)(a5 + 2) == 122 )
  {
    if ( (v5 & 0x10) != 0 )
    {
      v10 = *(_DWORD **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v10;
      if ( result > 3 )
      {
        *v10 = 2054448686;
        *(_QWORD *)(a4 + 32) += 4LL;
        return result;
      }
      v21 = 4;
      v22 = ".ftz";
      return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
    }
    return result;
  }
  if ( *(_WORD *)a5 != 24947 || *(_BYTE *)(a5 + 2) != 116 )
LABEL_52:
    BUG();
  if ( (v5 & 0x20) != 0 )
  {
    v9 = *(_DWORD **)(a4 + 32);
    result = *(_QWORD *)(a4 + 24) - (_QWORD)v9;
    if ( result <= 3 )
    {
      v21 = 4;
      v22 = ".sat";
      return sub_CB6200(a4, (unsigned __int8 *)v22, v21);
    }
    *v9 = 1952543534;
    *(_QWORD *)(a4 + 32) += 4LL;
  }
  return result;
}
