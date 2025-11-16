// Function: sub_398FCA0
// Address: 0x398fca0
//
void __fastcall sub_398FCA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  int v5; // eax

  if ( *(_DWORD *)(a1 + 4508) == 2 )
  {
    if ( *(_DWORD *)(a1 + 4508) == 1 )
    {
      nullsub_2030();
    }
    else
    {
      v4 = a1 + 4040;
      if ( *(_BYTE *)(a1 + 4513) )
        JUMPOUT(0x398FBB8);
      sub_39A1860(v4 + 192, *(_QWORD *)(a1 + 8), a2, a3);
      v5 = *(_DWORD *)(a1 + 4508);
      if ( v5 == 2 )
        JUMPOUT(0x398FBE8);
      if ( v5 == 3 )
        JUMPOUT(0x398FBC8);
    }
  }
}
