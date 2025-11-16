// Function: sub_AD0030
// Address: 0xad0030
//
void __fastcall sub_AD0030(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 *v4; // rdi

  v2 = 0;
  v3 = *(_QWORD *)(a1 + 16);
  while ( v3 )
  {
    v4 = *(__int64 **)(v3 + 24);
    if ( *(_BYTE *)v4 <= 0x15u && *(_BYTE *)v4 > 3u && (unsigned __int8)sub_ACFEF0(v4, 1) )
    {
      if ( v2 )
        v3 = *(_QWORD *)(v2 + 8);
      else
        v3 = *(_QWORD *)(a1 + 16);
    }
    else
    {
      v2 = v3;
      v3 = *(_QWORD *)(v3 + 8);
    }
  }
}
