// Function: sub_7CF9D0
// Address: 0x7cf9d0
//
__int64 __fastcall sub_7CF9D0(__int64 a1, int a2, int a3, __int64 a4)
{
  int v6; // r9d
  __int64 result; // rax
  unsigned __int8 v8; // dl
  char v9; // r10
  bool v10; // zf
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx

  if ( (a2 & 0xFFBFF468) != 0 )
    return 0;
  v6 = *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)sub_7CF970());
  result = *(_QWORD *)(a1 + 40);
  if ( !result )
    return 0;
  while ( 1 )
  {
    v8 = *(_BYTE *)(result + 82);
    if ( (v8 & 8) != 0
      && ((v8 & 0x10) != 0) == a3
      && *(_QWORD *)(result + 64) == a4
      && (a4 || *(_DWORD *)(result + 40) == unk_4F066A8)
      && (a2 & 1) == ((v8 & 0x20) != 0) )
    {
      v9 = *(_BYTE *)(result + 83);
      if ( (BYTE1(a2) & 1) == ((v9 & 2) != 0)
        && ((a2 & 4) != 0) == v8 >> 7
        && ((a2 & 0x200) != 0) == ((v9 & 8) != 0)
        && ((a2 & 0x800) != 0) == ((v9 & 4) != 0)
        && ((a2 & 2) != 0) == ((v8 & 0x40) != 0)
        && (a3 || v6 == *(_DWORD *)(result + 40)) )
      {
        break;
      }
    }
    result = *(_QWORD *)(result + 8);
    if ( !result )
      return 0;
  }
  v10 = *(_BYTE *)(result + 80) == 24;
  *(_DWORD *)(result + 44) = 0;
  if ( v10 )
  {
    v11 = *(unsigned __int8 *)(*(_QWORD *)(result + 88) + 80LL);
    if ( (unsigned __int8)v11 > 0x14u || (v12 = 1182720, !_bittest64(&v12, v11)) )
      *(_QWORD *)(result + 88) = 0;
  }
  return result;
}
