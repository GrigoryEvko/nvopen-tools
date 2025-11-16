// Function: sub_86BCA0
// Address: 0x86bca0
//
__int64 __fastcall sub_86BCA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rax

  v2 = a2;
  v3 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 488);
  if ( a2 != v3 )
  {
    do
    {
      if ( v3 != a1 )
      {
        if ( a1 == v2 )
          return v2;
        v4 = a1;
        while ( 1 )
        {
          v4 = sub_7340A0(*(_QWORD *)(v4 + 32));
          if ( *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 488) == v4 )
            break;
          if ( v4 == v2 )
            return v2;
        }
      }
      v2 = sub_7340A0(*(_QWORD *)(v2 + 32));
      v3 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 488);
    }
    while ( v3 != v2 );
  }
  return v2;
}
