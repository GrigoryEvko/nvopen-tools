// Function: sub_896740
// Address: 0x896740
//
__int64 __fastcall sub_896740(__int64 *a1, __int64 *a2, int a3)
{
  __int64 v4; // r12
  __int64 result; // rax
  __int64 v6; // rdx

  v4 = a1[12];
  result = sub_8788F0(*a2);
  if ( result )
  {
    v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(result + 88) + 168LL) + 152LL);
    if ( v6 )
    {
      if ( (*(_BYTE *)(v6 + 29) & 0x20) == 0 )
      {
        result = sub_883800(*(_QWORD *)(result + 96) + 192LL, *a1);
        if ( result )
        {
          while ( *(_BYTE *)(result + 80) != 6 || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(result + 96) + 32LL) + 64LL) != a3 )
          {
            result = *(_QWORD *)(result + 32);
            if ( !result )
              return result;
          }
          *(_QWORD *)(v4 + 24) = result;
        }
      }
    }
  }
  return result;
}
