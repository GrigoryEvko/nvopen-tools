// Function: sub_6F6C90
// Address: 0x6f6c90
//
__int64 __fastcall sub_6F6C90(__int64 a1, _DWORD *a2)
{
  __int64 v2; // rbx
  char v3; // al
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdx
  _DWORD *v8; // rax

  v2 = a1;
  v3 = *(_BYTE *)(a1 + 8);
  if ( v3 )
  {
    if ( v3 == 2 )
    {
      if ( (unsigned int)sub_6E5430() )
      {
        v8 = (_DWORD *)sub_6E1A20(a1);
        a1 = 2878;
        a2 = v8;
        sub_6851C0(0xB3Eu, v8);
      }
      result = sub_7305B0(a1, a2);
    }
    else
    {
      result = sub_6F6DD0();
    }
  }
  else
  {
    sub_6F6C80((_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL));
    result = sub_6F7150(*(_QWORD *)(a1 + 24) + 8LL, a2, v7);
  }
  v5 = *(_QWORD *)(result + 80);
  if ( v5 )
  {
    v6 = *(_QWORD *)(v2 + 16);
    if ( v6 )
      *(_QWORD *)(v5 + 128) = v6;
  }
  return result;
}
