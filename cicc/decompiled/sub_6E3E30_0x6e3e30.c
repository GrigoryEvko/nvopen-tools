// Function: sub_6E3E30
// Address: 0x6e3e30
//
__int64 __fastcall sub_6E3E30(__int64 a1, _DWORD *a2, __int64 *a3, _QWORD *a4)
{
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rbx
  char v7; // di
  __int64 v8; // rax
  bool v9; // zf

  *a2 = 0;
  result = 0;
  *a4 = 0;
  *a3 = 0;
  if ( *(_BYTE *)(a1 + 24) == 1 && (*(_BYTE *)(a1 + 27) & 2) != 0 && *(_BYTE *)(a1 + 56) == 3 )
  {
    v5 = *(_QWORD *)(a1 + 72);
    if ( *(_BYTE *)(v5 + 24) == 2 )
    {
      v6 = *(_QWORD *)(v5 + 56);
      v7 = *(_BYTE *)(v6 + 173);
      if ( v7 == 6 )
      {
        if ( *(_BYTE *)(v6 + 176) == 4 )
        {
          *a2 = 1;
          *a4 = *(_QWORD *)(v6 + 184);
          return 1;
        }
      }
      else if ( v7 == 12 && *(_BYTE *)(v6 + 176) == 8 )
      {
        v8 = sub_72F1F0(v6);
        *a3 = v8;
        v9 = v8 == 0;
        result = 1;
        *a2 = v9;
        if ( v9 )
          *a4 = *(_QWORD *)(v6 + 184);
      }
    }
  }
  return result;
}
