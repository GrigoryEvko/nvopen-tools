// Function: sub_14C8230
// Address: 0x14c8230
//
__int64 __fastcall sub_14C8230(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _DWORD *v4; // rdx

  result = a1;
  v4 = (_DWORD *)(a3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (_DWORD)a2 )
    v4 = *(_DWORD **)&v4[6 * ((unsigned int)(a2 - 1) - (unsigned __int64)(v4[5] & 0xFFFFFFF))];
  if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) == 15 )
  {
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)a1 = v4;
    *(_DWORD *)(a1 + 8) = HIDWORD(a2);
  }
  else
  {
    *(_BYTE *)(a1 + 16) = 0;
  }
  return result;
}
