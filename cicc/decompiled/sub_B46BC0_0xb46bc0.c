// Function: sub_B46BC0
// Address: 0xb46bc0
//
unsigned __int64 __fastcall sub_B46BC0(__int64 a1, char a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 result; // rax
  __int64 v4; // rcx

  if ( *(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL) != a1 + 24 )
  {
    v2 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v2 )
    {
      while ( 1 )
      {
        result = v2 - 24;
        if ( *(_BYTE *)result != 85 )
          return result;
        v4 = *(_QWORD *)(result - 32);
        if ( !v4 )
          return result;
        if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(result + 80) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
        {
          if ( (unsigned int)(*(_DWORD *)(v4 + 36) - 68) <= 3 )
            goto LABEL_13;
          if ( !a2 )
            return result;
        }
        else if ( !a2 )
        {
          return result;
        }
        if ( *(_BYTE *)v4
          || *(_QWORD *)(v4 + 24) != *(_QWORD *)(result + 80)
          || (*(_BYTE *)(v4 + 33) & 0x20) == 0
          || *(_DWORD *)(v4 + 36) != 291 )
        {
          return result;
        }
LABEL_13:
        if ( *(_QWORD *)(*(_QWORD *)(result + 40) + 56LL) != result + 24 )
        {
          v2 = *(_QWORD *)(result + 24) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v2 )
            continue;
        }
        return 0;
      }
    }
  }
  return 0;
}
