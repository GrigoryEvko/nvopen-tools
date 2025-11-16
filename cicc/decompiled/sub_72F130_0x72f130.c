// Function: sub_72F130
// Address: 0x72f130
//
__int64 __fastcall sub_72F130(_QWORD *a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  bool v3; // zf
  __int64 *v4; // rax

  v1 = a1[21];
  result = *(_QWORD *)(v1 + 152);
  if ( result )
  {
    if ( (*(_BYTE *)(result + 29) & 0x20) != 0 )
    {
      return 0;
    }
    else if ( (*(_BYTE *)(v1 + 109) & 0x40) != 0 )
    {
      v3 = *(_DWORD *)(result + 240) == -1;
      v4 = *(__int64 **)(*a1 + 96LL);
      if ( v3 )
        result = *v4;
      else
        result = v4[24];
      if ( result )
      {
        while ( *(_BYTE *)(result + 80) != 20
             || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(result + 88) + 176LL) + 206LL) & 2) == 0 )
        {
          result = *(_QWORD *)(result + 8);
          if ( !result )
            return result;
        }
        return *(_QWORD *)(*(_QWORD *)(result + 88) + 176LL);
      }
    }
    else
    {
      for ( result = *(_QWORD *)(result + 144); result; result = *(_QWORD *)(result + 112) )
      {
        if ( (*(_BYTE *)(result + 206) & 2) != 0 )
          break;
      }
    }
  }
  return result;
}
