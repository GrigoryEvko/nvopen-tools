// Function: sub_B921D0
// Address: 0xb921d0
//
__int64 __fastcall sub_B921D0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 v2; // dl
  __int64 v3; // rax

  v1 = sub_B92180(a1);
  if ( v1 )
  {
    v2 = *(_BYTE *)(v1 - 16);
    if ( (v2 & 2) != 0 )
    {
      v3 = *(_QWORD *)(*(_QWORD *)(v1 - 32) + 40LL);
      if ( v3 )
        return *(unsigned __int8 *)(v3 + 42);
    }
    else
    {
      v3 = *(_QWORD *)(v1 - 16 - 8LL * ((v2 >> 2) & 0xF) + 40);
      if ( v3 )
        return *(unsigned __int8 *)(v3 + 42);
    }
  }
  return 0;
}
