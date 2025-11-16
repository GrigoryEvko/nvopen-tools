// Function: sub_8752D0
// Address: 0x8752d0
//
__int64 __fastcall sub_8752D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  _QWORD *v3; // rcx
  __int64 v4; // rdx
  char v5; // cl

  result = qword_4F5FD70;
  if ( qword_4F5FD70 )
  {
    if ( dword_4F077C4 == 2 && !*(_QWORD *)(a1 + 88) )
    {
      v4 = qword_4F5FD70;
      do
      {
        v5 = *(_BYTE *)(v4 + 32);
        if ( v5 == 3 || v5 == 2 )
          *(_QWORD *)(*(_QWORD *)(v4 + 40) + 80LL) = 0;
        v4 = *(_QWORD *)v4;
      }
      while ( v4 );
    }
    v2 = qword_4F5FD68;
    v3 = *(_QWORD **)(result + 8);
    if ( v3 )
      *v3 = *(_QWORD *)qword_4F5FD68;
    else
      qword_4F5FD70 = *(_QWORD *)qword_4F5FD68;
    if ( *(_QWORD *)v2 )
      *(_QWORD *)(*(_QWORD *)v2 + 8LL) = *(_QWORD *)(result + 8);
    *(_QWORD *)v2 = qword_4F5FD60;
    qword_4F5FD60 = result;
    qword_4F5FD68 = 0;
    qword_4F5FD70 = 0;
  }
  return result;
}
