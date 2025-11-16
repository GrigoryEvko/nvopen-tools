// Function: sub_2E32770
// Address: 0x2e32770
//
__int64 __fastcall sub_2E32770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // r8
  int v7; // ecx
  int v8; // edi
  __int64 v9; // rcx

  result = sub_2E311E0(a1);
  v5 = *(_QWORD *)(a1 + 56);
  if ( v5 != result )
  {
    v6 = result;
    do
    {
      result = 2;
      v7 = *(_DWORD *)(v5 + 40) & 0xFFFFFF;
      v8 = v7 + 1;
      if ( v7 != 1 )
      {
        do
        {
          while ( 1 )
          {
            v9 = *(_QWORD *)(v5 + 32) + 40LL * (unsigned int)result;
            if ( a2 == *(_QWORD *)(v9 + 24) )
              break;
            result = (unsigned int)(result + 2);
            if ( v8 == (_DWORD)result )
              goto LABEL_8;
          }
          result = (unsigned int)(result + 2);
          *(_QWORD *)(v9 + 24) = a3;
        }
        while ( v8 != (_DWORD)result );
      }
LABEL_8:
      if ( (*(_BYTE *)v5 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
          v5 = *(_QWORD *)(v5 + 8);
      }
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v6 != v5 );
  }
  return result;
}
