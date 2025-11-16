// Function: sub_27CE4B0
// Address: 0x27ce4b0
//
char __fastcall sub_27CE4B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v5; // al
  char v6; // r13
  char result; // al
  int v8; // edx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rax

  v5 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 22 )
  {
    v6 = 0;
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) != 14 )
      return v6;
    return (unsigned int)sub_DF9B70(a3) != -1;
  }
  else
  {
    v6 = v5 == 5 || v5 > 0x1Cu;
    if ( !v6 )
      return v6;
    if ( v5 <= 0x1Cu )
      v8 = *(unsigned __int16 *)(a1 + 2);
    else
      v8 = v5 - 29;
    switch ( v8 )
    {
      case ' ':
      case '"':
      case '1':
      case '2':
      case '7':
      case '@':
        return v6;
      case '0':
        v6 = qword_4FFD608;
        if ( (_BYTE)qword_4FFD608 && (unsigned int)sub_DF9B70(a3) != -1 )
          return v6;
        result = sub_27CE300((unsigned __int8 *)a1, a2, a3);
        break;
      case '8':
        v6 = 0;
        if ( v5 == 85 )
        {
          v11 = *(_QWORD *)(a1 - 32);
          if ( v11 )
          {
            if ( !*(_BYTE *)v11 && *(_QWORD *)(v11 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v11 + 33) & 0x20) != 0 )
              return *(_DWORD *)(v11 + 36) == 299 || *(_DWORD *)(v11 + 36) == 8170;
          }
        }
        return v6;
      case '9':
        v9 = *(_QWORD *)(a1 + 8);
        v10 = *(unsigned __int8 *)(v9 + 8);
        if ( (unsigned int)(v10 - 17) <= 1 )
          LOBYTE(v10) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
        return (_BYTE)v10 == 14;
      default:
        return (unsigned int)sub_DF9B70(a3) != -1;
    }
  }
  return result;
}
