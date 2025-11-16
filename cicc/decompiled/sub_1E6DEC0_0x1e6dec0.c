// Function: sub_1E6DEC0
// Address: 0x1e6dec0
//
__int64 __fastcall sub_1E6DEC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  char v8; // al
  __int64 v9; // rdi
  __int64 result; // rax

  sub_1E6BD20(a1, a2, a3, a4, a5);
  v6 = *(_QWORD *)(a1 + 936);
  if ( v6 != a2 + 24 )
  {
    if ( !v6 )
      BUG();
    if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 46) & 8) != 0 )
    {
      do
        v6 = *(_QWORD *)(v6 + 8);
      while ( (*(_BYTE *)(v6 + 46) & 8) != 0 );
    }
    v6 = *(_QWORD *)(v6 + 8);
  }
  v7 = *(_QWORD *)(a1 + 2120);
  *(_QWORD *)(a1 + 2312) = v6;
  *(_DWORD *)(a1 + 2560) = 0;
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 40LL))(v7);
  v9 = *(_QWORD *)(a1 + 2120);
  *(_BYTE *)(a1 + 2568) = v8;
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 48LL))(v9);
  *(_BYTE *)(a1 + 2569) = result;
  return result;
}
