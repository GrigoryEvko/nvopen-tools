// Function: sub_6E5AE0
// Address: 0x6e5ae0
//
__int64 __fastcall sub_6E5AE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  _DWORD *v8; // r13
  __int64 v9; // [rsp+8h] [rbp-38h] BYREF
  __m128i v10[3]; // [rsp+10h] [rbp-30h] BYREF

  result = *(unsigned __int8 *)(a1 + 24);
  if ( (_BYTE)result == 1 )
  {
    if ( (*(_BYTE *)(a1 + 60) & 4) != 0 )
    {
      result = sub_6E5430();
      if ( (_DWORD)result )
      {
        result = sub_6E5AC0();
        if ( !(_DWORD)result )
        {
          v3 = (__int64 *)sub_7955B0(a1, v10);
          v9 = sub_724DC0(a1, v10, v4, v5, v6, v7);
          if ( (unsigned int)sub_7A39E0(a1, 1, v9, v10) )
            sub_721090(a1);
          v8 = sub_67E020(0xB75u, (_DWORD *)(a1 + 28), *v3);
          sub_67E370((__int64)v8, v10);
          sub_685910((__int64)v8, (FILE *)v10);
          result = sub_724E30(&v9);
        }
      }
      goto LABEL_12;
    }
  }
  else
  {
    switch ( (_BYTE)result )
    {
      case 0x24:
        *(_DWORD *)(a2 + 76) = 1;
        return result;
      case 0x14:
        result = *(_QWORD *)(a1 + 56);
        break;
      case 2:
        result = *(_QWORD *)(a1 + 56);
        if ( *(_BYTE *)(result + 173) != 7 || (*(_BYTE *)(result + 192) & 2) == 0 )
          return result;
        result = *(_QWORD *)(result + 200);
        break;
      default:
        return result;
    }
    if ( result && (*(_BYTE *)(result + 193) & 4) != 0 )
    {
      if ( (*(_BYTE *)(result + 206) & 0x10) == 0 )
      {
        result = sub_6E5430();
        if ( (_DWORD)result )
          result = sub_6851C0(0xB90u, (_DWORD *)(a1 + 28));
      }
LABEL_12:
      *(_DWORD *)(a2 + 72) = 1;
    }
  }
  return result;
}
