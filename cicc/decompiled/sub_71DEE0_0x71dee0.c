// Function: sub_71DEE0
// Address: 0x71dee0
//
__int64 __fastcall sub_71DEE0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx

  v4 = *(_QWORD *)(*(_QWORD *)a1 + 88LL);
  sub_7B8B50(a1, a2, a3, a4);
  sub_5F06F0(a1, a2, &dword_4F063F8, v5, v6);
  if ( (*(_BYTE *)(v4 + 206) & 8) != 0 && (*(_BYTE *)(v4 + 193) & 1) != 0 && *(_BYTE *)(v4 + 174) == 1 )
  {
    a2 = 1;
    if ( (unsigned int)sub_72F310(v4, 1, v7, v8, v9, v10) )
      sub_600530(*(_QWORD *)(*(_QWORD *)(v4 + 40) + 32LL));
  }
  sub_71D150(v4, a2, v7, v8, v9, v10);
  return sub_7B8B50(v4, a2, v11, v12);
}
