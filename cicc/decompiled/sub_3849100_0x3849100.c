// Function: sub_3849100
// Address: 0x3849100
//
__int64 __fastcall sub_3849100(__int64 a1, unsigned __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  __int16 v14; // [rsp+10h] [rbp-40h] BYREF
  __int64 v15; // [rsp+18h] [rbp-38h]

  v7 = sub_3761980(a1, a2, a3);
  v9 = v8;
  v10 = *(_QWORD *)(v7 + 48) + 16LL * (unsigned int)v8;
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v14 = v11;
  v15 = v12;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 17) > 0xD3u )
    {
      if ( (unsigned __int16)(v11 - 2) <= 7u || (unsigned __int16)(v11 - 176) <= 0x1Fu )
        return sub_375E510(a1, v7, v9, a4, a5);
      return sub_375E6F0(a1, v7, v9, a4, a5);
    }
  }
  else if ( !sub_30070B0((__int64)&v14) )
  {
    if ( sub_3007070((__int64)&v14) )
      return sub_375E510(a1, v7, v9, a4, a5);
    return sub_375E6F0(a1, v7, v9, a4, a5);
  }
  return sub_375E8D0(a1, v7, v9, a4, a5);
}
