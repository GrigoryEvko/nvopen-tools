// Function: sub_1F3A680
// Address: 0x1f3a680
//
__int64 __fastcall sub_1F3A680(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  int v11; // edx

  result = sub_1F4B690(a2);
  if ( (_BYTE)result )
  {
    v7 = *(_QWORD *)(a2 + 168);
    if ( v7
      && (v8 = v7 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(a3 + 16) + 6LL),
          v9 = (unsigned int)*(unsigned __int16 *)(v8 + 6) + a4,
          *(unsigned __int16 *)(v8 + 8) > (unsigned int)v9) )
    {
      v10 = *(_QWORD *)(a2 + 152);
      v11 = *(_DWORD *)(v10 + 4 * v9);
      LOBYTE(v10) = v11 <= 1;
      LOBYTE(v11) = v11 != -1;
      return v11 & (unsigned int)v10;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
