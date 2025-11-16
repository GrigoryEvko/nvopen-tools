// Function: sub_7E0220
// Address: 0x7e0220
//
__int64 __fastcall sub_7E0220(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rax

  v1 = a1;
  if ( (*(_BYTE *)(a1 + 96) & 2) == 0 )
  {
    if ( *(_QWORD *)(a1 + 128) != -1 )
    {
      v1 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 112) + 8LL) + 16LL);
      if ( (*(_BYTE *)(v1 + 96) & 2) != 0 )
      {
        while ( 1 )
        {
          v5 = *(_QWORD *)(v1 + 40);
          if ( (*(_BYTE *)(v5 + 176) & 0x50) != 0 )
            break;
          v3 = *(_QWORD *)(*(_QWORD *)(v5 + 168) + 24LL);
          if ( !v3 )
            break;
          v4 = sub_8E5650(v3);
          if ( a1 == v4 )
            return v1;
          v1 = v4;
        }
      }
    }
    return 0;
  }
  return v1;
}
