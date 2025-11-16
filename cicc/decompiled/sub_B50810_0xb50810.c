// Function: sub_B50810
// Address: 0xb50810
//
__int64 __fastcall sub_B50810(
        unsigned int a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned __int8 v10; // dl
  int v11; // eax
  char v12; // dl
  int v13; // edx
  char v14; // si
  __int64 result; // rax
  unsigned int v16; // r12d
  unsigned int v17; // ebx
  int v18; // eax
  unsigned int v19; // r15d
  unsigned int v20; // edx
  unsigned int v21; // ebx
  __int64 v22; // [rsp+8h] [rbp-28h]
  __int64 v23; // [rsp+8h] [rbp-28h]

  if ( a1 != 49 )
    goto LABEL_2;
  v13 = *(unsigned __int8 *)(a4 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 > 1 )
  {
    v14 = 0;
    if ( v13 == 17 )
      goto LABEL_12;
  }
  else
  {
    v14 = 1;
    if ( v13 == 17 )
    {
      if ( a2 != 49 )
        goto LABEL_3;
      goto LABEL_5;
    }
  }
  if ( (v13 == 18) == v14 )
  {
LABEL_2:
    if ( a2 != 49 )
      goto LABEL_3;
    v10 = *(_BYTE *)(a5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 <= 1 )
    {
LABEL_5:
      v11 = *(unsigned __int8 *)(a5 + 8);
      if ( v11 == 17 )
        goto LABEL_3;
      v12 = 1;
LABEL_7:
      if ( (v11 == 18) == v12 )
        goto LABEL_3;
      goto LABEL_12;
    }
    v11 = v10;
    if ( v10 != 17 )
    {
      v12 = 0;
      goto LABEL_7;
    }
  }
LABEL_12:
  if ( a1 != 49 || a2 != 49 )
    return 0;
LABEL_3:
  switch ( byte_3F2B3E0[13 * (a1 - 38) + a2 - 38] )
  {
    case 0:
      return 0;
    case 1:
    case 0xF:
      return a1;
    case 2:
    case 0xE:
    case 0x10:
      return a2;
    case 3:
      result = 0;
      if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 > 1 && *(_BYTE *)(a5 + 8) == 12 )
        return a1;
      return result;
    case 4:
      result = 0;
      if ( a4 == a5 )
        return a1;
      return result;
    case 5:
      result = 0;
      if ( *(_BYTE *)(a3 + 8) == 12 )
        return a2;
      return result;
    case 7:
      result = 0;
      if ( byte_4F818A8 )
        return result;
      if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
        a3 = **(_QWORD **)(a3 + 16);
      if ( (unsigned int)*(unsigned __int8 *)(a5 + 8) - 17 <= 1 )
        a5 = **(_QWORD **)(a5 + 16);
      result = 0;
      if ( *(_DWORD *)(a5 + 8) >> 8 != *(_DWORD *)(a3 + 8) >> 8 )
        return result;
      v21 = sub_BCB060(a4);
      if ( v21 == 64 )
        return 49;
      if ( !a6 || a6 != a8 )
        return 0;
      result = v21 >= (unsigned int)sub_BCB060(a6) ? 0x31 : 0;
      break;
    case 8:
      v23 = a5;
      v19 = sub_BCB060(a3);
      v20 = sub_BCB060(v23);
      if ( a3 == v23 )
        return 49;
      result = a1;
      if ( v19 >= v20 )
      {
        result = 0;
        if ( v19 > v20 )
          return a2;
      }
      return result;
    case 9:
      return 39;
    case 0xB:
      v22 = a5;
      if ( a7 )
      {
        v16 = sub_BCB060(a7);
        v17 = sub_BCB060(a3);
        v18 = sub_BCB060(v22);
        if ( v16 >= v17 && v17 == v18 )
          return 49;
      }
      return 0;
    case 0xC:
      if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
        a3 = **(_QWORD **)(a3 + 16);
      if ( (unsigned int)*(unsigned __int8 *)(a5 + 8) - 17 <= 1 )
        a5 = **(_QWORD **)(a5 + 16);
      return (unsigned int)(*(_DWORD *)(a5 + 8) >> 8 != *(_DWORD *)(a3 + 8) >> 8) + 49;
    case 0xD:
      return 50;
    case 0x11:
      return 43;
    default:
      BUG();
  }
  return result;
}
