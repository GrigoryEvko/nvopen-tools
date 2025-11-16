// Function: sub_728520
// Address: 0x728520
//
__int64 __fastcall sub_728520(_QWORD *a1, char a2, __int64 a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  __int64 result; // rax
  __int64 v7; // r14

  v4 = (_QWORD *)sub_727DD0(a2, a1);
  v4[1] = a3;
  v5 = v4;
  *v4 = a1[15];
  a1[15] = v4;
  result = dword_4F07590;
  if ( !dword_4F07590 && *((_BYTE *)v5 + 16) == 3 )
  {
    v7 = *(_QWORD *)(a3 + 160);
    result = sub_8D3A70(v7);
    if ( !(_DWORD)result || (*(_BYTE *)(v7 + 177) & 0x20) != 0 )
    {
      *((_BYTE *)v5 + 17) = 1;
      result = (__int64)sub_728460((__int64)a1, unk_4D03FF0);
    }
  }
  if ( (*(_BYTE *)(a3 - 8) & 2) != 0 && (*(_BYTE *)(a1 - 1) & 2) == 0 )
  {
    *((_BYTE *)v5 + 17) = 1;
    result = (__int64)sub_728460((__int64)a1, (__int64)qword_4D03FD0);
  }
  if ( !*((_BYTE *)v5 + 17) )
  {
    result = sub_7607C0(a1, 6);
    if ( *((char *)a1 - 8) < 0 )
    {
      *((_BYTE *)a1 - 8) &= ~0x80u;
      return sub_75B260(a1, 6);
    }
  }
  return result;
}
