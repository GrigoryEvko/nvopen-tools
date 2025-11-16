// Function: sub_14C43E0
// Address: 0x14c43e0
//
__int64 __fastcall sub_14C43E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // rax
  unsigned int v9; // [rsp+Ch] [rbp-34h]

  if ( *(_BYTE *)(a1 + 16) != 56 )
    return a1;
  v5 = (unsigned int)sub_14C3BA0(a1);
  v9 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v9 )
  {
    v6 = v9;
    v7 = 0;
    do
    {
      if ( (_DWORD)v5 != (_DWORD)v7 )
      {
        v8 = sub_146F1B0(a2, *(_QWORD *)(a1 + 24 * (v7 - v6)));
        if ( !sub_146CEE0(a2, v8, a3) )
          return a1;
        v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
      }
      ++v7;
    }
    while ( v9 != (_DWORD)v7 );
  }
  else
  {
    v6 = 0;
  }
  return *(_QWORD *)(a1 + 24 * (v5 - v6));
}
