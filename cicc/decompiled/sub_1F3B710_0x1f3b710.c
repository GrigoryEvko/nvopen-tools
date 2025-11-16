// Function: sub_1F3B710
// Address: 0x1f3b710
//
__int64 __fastcall sub_1F3B710(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // rax
  int v8; // ecx

  v4 = *(_QWORD *)(a2 + 24);
  v5 = *(_QWORD *)(*(_QWORD *)(v4 + 56) + 40LL);
  v6 = sub_1E69D60(v5, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v7 = sub_1E69D60(v5, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v8 = **(unsigned __int16 **)(a2 + 16);
  if ( **(_WORD **)(v6 + 16) != **(_WORD **)(a2 + 16) && v8 == **(unsigned __int16 **)(v7 + 16) )
  {
    v6 = v7;
    *a3 = 1;
    if ( v8 != **(unsigned __int16 **)(v7 + 16) )
      return 0;
  }
  else
  {
    *a3 = 0;
    if ( v8 != **(unsigned __int16 **)(v6 + 16) )
      return 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 464LL))(a1, v6, v4) )
    return 0;
  return sub_1E69E00(v5, *(_DWORD *)(*(_QWORD *)(v6 + 32) + 8LL));
}
