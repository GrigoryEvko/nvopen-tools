// Function: sub_2E21C60
// Address: 0x2e21c60
//
__int64 __fastcall sub_2E21C60(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v6; // rsi
  __int64 v7; // r10
  signed __int64 v8; // rcx
  __int64 v9; // rcx
  unsigned int v10; // ecx
  __int64 v11; // rdx

  result = sub_2EBFBC0(a2);
  if ( result )
  {
    v5 = result;
    v6 = *(unsigned __int16 *)result;
    while ( (_WORD)v6 )
    {
      v7 = *(_QWORD *)(a3 + 104);
      result = *(_QWORD *)(a3 + 96);
      v8 = 0xAAAAAAAAAAAAAAABLL * ((v7 - result) >> 2);
      if ( v8 >> 2 > 0 )
      {
        v9 = result + 48 * (v8 >> 2);
        while ( (unsigned __int16)v6 != *(_DWORD *)result )
        {
          if ( (unsigned __int16)v6 == *(_DWORD *)(result + 12) )
          {
            result += 12;
            goto LABEL_10;
          }
          if ( (unsigned __int16)v6 == *(_DWORD *)(result + 24) )
          {
            result += 24;
            goto LABEL_10;
          }
          if ( (unsigned __int16)v6 == *(_DWORD *)(result + 36) )
          {
            result += 36;
            goto LABEL_10;
          }
          result += 48;
          if ( result == v9 )
          {
            v8 = 0xAAAAAAAAAAAAAAABLL * ((v7 - result) >> 2);
            goto LABEL_18;
          }
        }
        goto LABEL_10;
      }
LABEL_18:
      if ( v8 != 2 )
      {
        if ( v8 != 3 )
        {
          if ( v8 != 1 || (unsigned __int16)v6 != *(_DWORD *)result )
            goto LABEL_12;
          goto LABEL_10;
        }
        if ( (unsigned __int16)v6 == *(_DWORD *)result )
          goto LABEL_10;
        result += 12;
      }
      if ( (unsigned __int16)v6 != *(_DWORD *)result )
      {
        result += 12;
        if ( (unsigned __int16)v6 != *(_DWORD *)result )
        {
LABEL_12:
          result = *(_DWORD *)(*(_QWORD *)(*a1 + 8LL) + 24 * v6 + 16) >> 12;
          v10 = *(_DWORD *)(*(_QWORD *)(*a1 + 8LL) + 24 * v6 + 16) & 0xFFF;
          v11 = *(_QWORD *)(*a1 + 56LL) + 2 * result;
          do
          {
            if ( !v11 )
              break;
            v11 += 2;
            result = v10 >> 6;
            *(_QWORD *)(a1[1] + 8 * result) |= 1LL << v10;
            v10 += *(__int16 *)(v11 - 2);
          }
          while ( *(_WORD *)(v11 - 2) );
          goto LABEL_15;
        }
      }
LABEL_10:
      if ( result == v7 || *(_BYTE *)(result + 8) )
        goto LABEL_12;
LABEL_15:
      v6 = *(unsigned __int16 *)(v5 + 2);
      v5 += 2;
    }
  }
  return result;
}
