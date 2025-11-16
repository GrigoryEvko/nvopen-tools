// Function: sub_E0F930
// Address: 0xe0f930
//
__int64 __fastcall sub_E0F930(_QWORD *a1, __int64 a2)
{
  _BYTE *v3; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int16 v13; // si
  __int16 v14; // ax

  v3 = (_BYTE *)*a1;
  if ( a1[1] != *a1 )
  {
    while ( *v3 == 66 )
    {
      *a1 = v3 + 1;
      v9 = sub_E0F8D0((__int64)a1);
      v10 = v5;
      if ( !v9 )
        return 0;
      v11 = sub_E0E790((__int64)(a1 + 102), 40, v5, v6, v7, v8);
      v12 = v11;
      if ( !v11 )
        return 0;
      v13 = *(_WORD *)(a2 + 9);
      *(_WORD *)(v11 + 8) = *(_WORD *)(v11 + 8) & 0xC000 | 9;
      v14 = *(_WORD *)(v11 + 9);
      *(_QWORD *)(v12 + 16) = a2;
      a2 = v12;
      *(_QWORD *)(v12 + 24) = v9;
      *(_QWORD *)(v12 + 32) = v10;
      *(_WORD *)(v12 + 9) = v13 & 0xFC0 | v14 & 0xF03F;
      *(_QWORD *)v12 = &unk_49DF128;
      v3 = (_BYTE *)*a1;
      if ( *a1 == a1[1] )
        return a2;
    }
  }
  return a2;
}
