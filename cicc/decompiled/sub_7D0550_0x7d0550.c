// Function: sub_7D0550
// Address: 0x7d0550
//
__int64 __fastcall sub_7D0550(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  char v6; // r9
  __int64 v7; // rdi
  __int64 v8; // r10
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // rsi

  result = 1;
  if ( a1 == a2 )
    return result;
  v5 = *(unsigned __int8 *)(a1 + 80);
  v6 = *(_BYTE *)(a2 + 80);
  result = 0;
  if ( (_BYTE)v5 != v6 )
    return result;
  switch ( (_BYTE)v5 )
  {
    case 7:
      return *(_QWORD *)(a1 + 88) == *(_QWORD *)(a2 + 88);
    case 0x17:
      v9 = *(_QWORD *)(a1 + 88);
      v10 = *(_QWORD *)(a2 + 88);
      if ( (*(_BYTE *)(v9 + 124) & 1) != 0 )
        v9 = sub_735B70(*(_QWORD *)(a1 + 88));
      if ( (*(_BYTE *)(v10 + 124) & 1) != 0 )
        v10 = sub_735B70(v10);
      return v9 == v10;
    case 0xB:
      v11 = *(_QWORD *)(a1 + 88);
      v12 = *(_QWORD *)(a2 + 88);
      result = v11 == v12;
      if ( v11 != v12
        && a3
        && dword_4F077BC
        && !(_DWORD)qword_4F077B4
        && (*(_BYTE *)(v11 + 88) & 0x70) == 0x30
        && (*(_BYTE *)(v12 + 88) & 0x70) == 0x30 )
      {
        return 1;
      }
      break;
    default:
      result = dword_4D04208;
      if ( !dword_4D04208 )
      {
        a4 &= 0x20000u;
        if ( (_DWORD)a4 )
          return result;
      }
      if ( (_BYTE)v5 == 3 )
      {
        v7 = *(_QWORD *)(a1 + 88);
      }
      else
      {
        result = 0;
        if ( dword_4F077C4 != 2 || (unsigned __int8)(v5 - 4) > 2u )
          return result;
        v7 = *(_QWORD *)(a1 + 88);
        v8 = *(_QWORD *)(a2 + 88);
        if ( (_BYTE)v5 == 6 || v6 != 3 )
        {
LABEL_13:
          result = 1;
          if ( v8 != v7 )
            return (unsigned int)sub_8D97D0(v7, v8, 0, a4, v5) != 0;
          return result;
        }
      }
      v8 = *(_QWORD *)(a2 + 88);
      goto LABEL_13;
  }
  return result;
}
