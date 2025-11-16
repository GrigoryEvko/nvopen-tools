// Function: sub_CF6FD0
// Address: 0xcf6fd0
//
__int64 __fastcall sub_CF6FD0(unsigned __int8 *a1)
{
  __int64 v1; // rbp
  int v2; // eax
  __int64 result; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD v7[4]; // [rsp-20h] [rbp-20h] BYREF

  v2 = *a1;
  if ( (unsigned __int8)v2 <= 0x1Cu )
    return 0;
  v4 = (unsigned int)(v2 - 34);
  if ( (unsigned __int8)v4 > 0x33u )
    return 0;
  v5 = 0x8000000000041LL;
  if ( !_bittest64(&v5, v4) )
    return 0;
  v7[3] = v1;
  result = sub_A74710((_QWORD *)a1 + 9, 0, 22);
  if ( !(_BYTE)result )
  {
    v6 = *((_QWORD *)a1 - 4);
    if ( v6 && !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == *((_QWORD *)a1 + 10) )
    {
      v7[0] = *(_QWORD *)(v6 + 120);
      return sub_A74710(v7, 0, 22);
    }
    else
    {
      return 0;
    }
  }
  return result;
}
