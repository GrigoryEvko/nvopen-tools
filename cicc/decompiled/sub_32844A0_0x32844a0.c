// Function: sub_32844A0
// Address: 0x32844a0
//
__int64 __fastcall sub_32844A0(unsigned __int16 *a1, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // rdx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int16 v8; // [rsp+0h] [rbp-30h] BYREF
  __int64 v9; // [rsp+8h] [rbp-28h]

  v2 = *a1;
  if ( (_WORD)v2 )
  {
    if ( (unsigned __int16)(v2 - 17) <= 0xD3u )
    {
      v9 = 0;
      LOWORD(v2) = word_4456580[v2 - 1];
      v8 = v2;
      if ( !(_WORD)v2 )
        return sub_3007260((__int64)&v8);
      goto LABEL_7;
    }
    goto LABEL_3;
  }
  if ( !sub_30070B0((__int64)a1) )
  {
LABEL_3:
    v3 = *((_QWORD *)a1 + 1);
    goto LABEL_4;
  }
  LOWORD(v2) = sub_3009970((__int64)a1, a2, v5, v6, v7);
LABEL_4:
  v8 = v2;
  v9 = v3;
  if ( !(_WORD)v2 )
    return sub_3007260((__int64)&v8);
LABEL_7:
  if ( (_WORD)v2 == 1 || (unsigned __int16)(v2 - 504) <= 7u )
    BUG();
  return *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v2 - 16];
}
