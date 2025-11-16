// Function: sub_2AB37C0
// Address: 0x2ab37c0
//
__int64 __fastcall sub_2AB37C0(__int64 a1, unsigned __int8 *a2)
{
  int v2; // eax
  unsigned __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // [rsp-8h] [rbp-18h]

  if ( (!*(_BYTE *)(a1 + 108) || !*(_DWORD *)(a1 + 100))
    && !(unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *((_QWORD *)a2 + 5))
    || sub_991A70(a2, 0, 0, 0, 0, 1u, 0) )
  {
    return 0;
  }
  v2 = *a2;
  if ( (unsigned __int8)(v2 - 61) <= 1u || (_BYTE)v2 == 85 )
  {
    if ( !(unsigned __int8)sub_B19060(*(_QWORD *)(a1 + 440) + 440LL, (__int64)a2, (unsigned int)(v2 - 61), v13) )
      return 0;
    v2 = *a2;
  }
  v3 = (unsigned int)(v2 - 31);
  if ( (unsigned __int8)v3 <= 0x35u )
  {
    v5 = 0x20000020000003LL;
    if ( _bittest64(&v5, v3) )
      return 0;
  }
  if ( (unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *((_QWORD *)a2 + 5)) )
    return 1;
  switch ( *a2 )
  {
    case '0':
    case '1':
    case '3':
    case '4':
      v6 = sub_986520((__int64)a2);
      return (unsigned int)sub_D48480(*(_QWORD *)(a1 + 416), *(_QWORD *)(v6 + 32), v7, v8) ^ 1;
    case '=':
      v12 = sub_228AED0(a2);
      return (unsigned int)sub_31A5290(*(_QWORD *)(a1 + 440), v12) ^ 1;
    case '>':
      v9 = sub_228AED0(a2);
      if ( !(unsigned __int8)sub_31A5290(*(_QWORD *)(a1 + 440), v9) )
        return 1;
      result = (unsigned int)sub_D48480(*(_QWORD *)(a1 + 416), *((_QWORD *)a2 - 8), v10, v11) ^ 1;
      break;
    case 'U':
      return 1;
    default:
      BUG();
  }
  return result;
}
