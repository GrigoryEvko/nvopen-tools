// Function: sub_33C9580
// Address: 0x33c9580
//
__int64 __fastcall sub_33C9580(__int64 a1, unsigned int a2)
{
  unsigned __int16 *v2; // rsi
  int v3; // eax
  __int64 v4; // rbx
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int16 v10; // [rsp+0h] [rbp-40h] BYREF
  __int64 v11; // [rsp+8h] [rbp-38h]
  __int16 v12; // [rsp+10h] [rbp-30h] BYREF
  __int64 v13; // [rsp+18h] [rbp-28h]

  v2 = (unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * a2);
  v3 = *v2;
  v4 = *((_QWORD *)v2 + 1);
  v10 = v3;
  v11 = v4;
  if ( (_WORD)v3 )
  {
    if ( (unsigned __int16)(v3 - 17) > 0xD3u )
    {
      v12 = v3;
      v13 = v4;
      goto LABEL_4;
    }
    LOWORD(v3) = word_4456580[v3 - 1];
    v6 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v10) )
    {
      v13 = v4;
      v12 = 0;
      return sub_3007260((__int64)&v12);
    }
    LOWORD(v3) = sub_3009970((__int64)&v10, (__int64)v2, v7, v8, v9);
  }
  v12 = v3;
  v13 = v6;
  if ( !(_WORD)v3 )
    return sub_3007260((__int64)&v12);
LABEL_4:
  if ( (_WORD)v3 == 1 || (unsigned __int16)(v3 - 504) <= 7u )
    BUG();
  return *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v3 - 16];
}
