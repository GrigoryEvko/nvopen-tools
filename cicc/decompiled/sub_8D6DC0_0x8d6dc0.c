// Function: sub_8D6DC0
// Address: 0x8d6dc0
//
unsigned __int64 __fastcall sub_8D6DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 result; // rax
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rdx
  char v10; // al
  __int64 v11; // rdi
  __int64 v12; // rdx
  bool v13; // zf

  result = *(unsigned __int8 *)(a1 + 24);
  if ( (*(_BYTE *)(a1 + 25) & 3) != 0 )
  {
    switch ( (char)result )
    {
      case 0:
      case 11:
      case 20:
      case 23:
        break;
      case 1:
        v6 = *(unsigned __int8 *)(a1 + 56);
        if ( (unsigned __int8)(v6 - 94) <= 1u )
        {
          result = *(unsigned int *)(a2 + 72);
          if ( (_DWORD)result )
            return result;
          result = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 56LL);
          v8 = *(_QWORD *)(result + 120);
          goto LABEL_21;
        }
        result = (unsigned int)(v6 - 71);
        if ( (unsigned __int8)result > 0x20u )
          return result;
        v7 = 0x106000003LL;
        if ( !_bittest64(&v7, result) )
          return result;
        break;
      case 2:
        result = *(_QWORD *)(a1 + 56);
        if ( *(_BYTE *)(result + 173) != 2 )
          return result;
        v8 = *(_QWORD *)(result + 128);
        result = *(unsigned int *)(a2 + 72);
        if ( (_DWORD)result )
          return result;
        goto LABEL_11;
      case 3:
        v8 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 120LL);
        result = *(unsigned int *)(a2 + 72);
        if ( (_DWORD)result )
          return result;
        goto LABEL_11;
      case 5:
      case 24:
        goto LABEL_10;
      case 10:
        return result;
      default:
        goto LABEL_18;
    }
LABEL_7:
    result = *(unsigned int *)(a2 + 72);
    if ( !(_DWORD)result )
LABEL_8:
      *(_DWORD *)(a2 + 76) = 1;
  }
  else
  {
    switch ( (char)result )
    {
      case 0:
      case 5:
      case 11:
      case 16:
      case 17:
      case 19:
      case 20:
      case 23:
      case 24:
        goto LABEL_7;
      case 1:
        if ( *(_BYTE *)(a1 + 56) != 21 )
          return result;
        v11 = *(_QWORD *)(a1 + 72);
        if ( (*(_BYTE *)(v11 + 25) & 3) == 0 )
          goto LABEL_7;
        sub_76CDC0((_QWORD *)v11, a2, a3, a4, a5);
        v8 = *(_QWORD *)(a2 + 136);
        if ( !v8 )
          goto LABEL_7;
        result = sub_8D3410(*(_QWORD *)(a2 + 136));
        if ( (_DWORD)result )
        {
          result = sub_8D40F0(v8);
          v8 = result;
          v13 = *(_DWORD *)(a2 + 72) == 0;
          *(_QWORD *)(a2 + 136) = result;
          if ( v13 )
            goto LABEL_21;
        }
        else if ( !*(_DWORD *)(a2 + 72) )
        {
          goto LABEL_12;
        }
        return result;
      case 2:
        result = sub_8D6D50(*(_QWORD *)(a1 + 56));
        v8 = result;
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        goto LABEL_11;
      case 3:
        if ( !*(_DWORD *)(a2 + 144) )
          return result;
        result = *(_QWORD *)(a1 + 56);
        if ( *(_QWORD *)(result + 8) )
          return result;
        if ( *(char *)(result + 169) >= 0 )
          return result;
        if ( !qword_4F04C50 )
          return result;
        if ( *(_QWORD *)(qword_4F04C50 + 64LL) != result )
          return result;
        v12 = *(_QWORD *)(qword_4F04C50 + 32LL);
        result = (unsigned int)*(unsigned __int8 *)(v12 + 174) - 1;
        if ( (unsigned __int8)(*(_BYTE *)(v12 + 174) - 1) > 1u )
          return result;
        result = *(_QWORD *)(v12 + 40);
        v8 = *(_QWORD *)(result + 32);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        goto LABEL_11;
      case 6:
LABEL_10:
        v8 = *(_QWORD *)a1;
        if ( !*(_DWORD *)(a2 + 72) )
          goto LABEL_11;
        return result;
      case 7:
        result = *(_QWORD *)(a1 + 56);
        if ( (*(_BYTE *)result & 1) == 0 )
          return result;
        v8 = *(_QWORD *)(result + 8);
        result = sub_8D3410(v8);
        if ( (_DWORD)result )
        {
          result = sub_8D40F0(v8);
          v8 = result;
          if ( *(_DWORD *)(a2 + 72) )
            return result;
        }
        else if ( *(_DWORD *)(a2 + 72) )
        {
          return result;
        }
LABEL_11:
        if ( !v8 )
          return result;
        goto LABEL_12;
      case 10:
        return result;
      case 18:
        v9 = *(_QWORD *)(a1 + 56);
        v10 = *(_BYTE *)(v9 + 48);
        if ( v10 == 2 )
        {
          result = sub_8D6D50(*(_QWORD *)(v9 + 56));
          v8 = result;
          if ( *(_DWORD *)(a2 + 72) )
            return result;
LABEL_21:
          if ( v8 )
          {
LABEL_12:
            *(_QWORD *)(a2 + 136) = v8;
            *(_DWORD *)(a2 + 72) = 1;
            return result;
          }
          goto LABEL_8;
        }
        if ( (unsigned __int8)(v10 - 3) > 1u )
          goto LABEL_7;
        result = sub_76CDC0(*(_QWORD **)(v9 + 56), a2, v9, a4, a5);
        if ( !*(_DWORD *)(a2 + 72) )
          goto LABEL_8;
        break;
      default:
LABEL_18:
        sub_721090();
    }
  }
  return result;
}
