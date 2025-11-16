// Function: sub_865840
// Address: 0x865840
//
__int64 __fastcall sub_865840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, unsigned int a7)
{
  __int64 result; // rax
  int v8; // ebx
  __int64 v9; // [rsp-10h] [rbp-60h]
  _DWORD v14[9]; // [rsp+2Ch] [rbp-24h] BYREF

  if ( *(_BYTE *)(a5 + 80) == 19 && (*(_BYTE *)(*(_QWORD *)(a5 + 88) + 266LL) & 1) != 0 )
  {
    v14[0] = 0;
    v8 = sub_85EBD0(a5, v14);
    sub_85E1C0(a1, a2, a3, a4, a5, a6, a7);
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    *(_DWORD *)(result + 552) = v8;
  }
  else
  {
    sub_864700(a1, a2, a3, a4, a5, a6, 1, a7);
    return v9;
  }
  return result;
}
