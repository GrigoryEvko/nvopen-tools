// Function: sub_6FB450
// Address: 0x6fb450
//
__int64 __fastcall sub_6FB450(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9

  if ( *(_BYTE *)(a1 + 17) == 2 || (result = sub_6ED0A0(a1), (_DWORD)result) )
  {
    result = sub_8D3410(*(_QWORD *)a1);
    if ( (_DWORD)result )
    {
      if ( dword_4D04964 )
      {
        sub_6E5C80(unk_4F07471, 0x128u, (_DWORD *)(a1 + 68));
        return sub_6FB030(a1, 296, v7, v8, v9, v10);
      }
      else
      {
        return sub_6FB030(a1, a2, v3, v4, v5, v6);
      }
    }
  }
  return result;
}
