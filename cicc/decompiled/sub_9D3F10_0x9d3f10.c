// Function: sub_9D3F10
// Address: 0x9d3f10
//
void __fastcall sub_9D3F10(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi

  v3 = a1[1];
  if ( v3 == a1[2] )
  {
    sub_9D3C80(a1, v3, a2);
  }
  else
  {
    if ( v3 )
    {
      *(_BYTE *)v3 = *(_BYTE *)a2;
      *(_QWORD *)(v3 + 8) = v3 + 24;
      *(_QWORD *)(v3 + 16) = 0xC00000000LL;
      if ( *(_DWORD *)(a2 + 16) )
        sub_9C31C0(v3 + 8, (char **)(a2 + 8));
      v3 = a1[1];
    }
    a1[1] = v3 + 72;
  }
}
