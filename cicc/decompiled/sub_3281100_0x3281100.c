// Function: sub_3281100
// Address: 0x3281100
//
__int16 __fastcall sub_3281100(unsigned __int16 *a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // rax
  __int16 v4; // cx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8

  v2 = *a1;
  if ( !(_WORD)v2 )
  {
    if ( !sub_30070B0((__int64)a1) )
      return *(_QWORD *)a1;
    v4 = sub_3009970((__int64)a1, a2, v5, v6, v7);
LABEL_5:
    LOWORD(v3) = v4;
    return v3;
  }
  if ( (unsigned __int16)(v2 - 17) <= 0xD3u )
  {
    v4 = word_4456580[v2 - 1];
    goto LABEL_5;
  }
  return *(_QWORD *)a1;
}
