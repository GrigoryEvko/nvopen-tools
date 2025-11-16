// Function: sub_1F3A880
// Address: 0x1f3a880
//
char __fastcall sub_1F3A880(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int16 v6; // ax
  __int64 v7; // rax
  __int64 *v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 (*v11)(void); // rdx
  __int64 (*v12)(); // rax
  __int64 v13; // r8
  __int64 v14; // rax

  v4 = a2;
  v6 = *(_WORD *)(a2 + 46);
  if ( (v6 & 4) == 0 && (v6 & 8) != 0 )
  {
    a2 = 64;
    LOBYTE(v7) = sub_1E15D00(v4, 0x40u, 1);
  }
  else
  {
    v7 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 6) & 1LL;
  }
  if ( !(_BYTE)v7 )
  {
    LOBYTE(v7) = 1;
    if ( (unsigned __int16)(**(_WORD **)(v4 + 16) - 2) > 3u )
    {
      v8 = *(__int64 **)(a4 + 16);
      v9 = 0;
      v10 = *v8;
      v11 = *(__int64 (**)(void))(*v8 + 56);
      if ( v11 != sub_1D12D20 )
      {
        v14 = v11();
        v8 = *(__int64 **)(a4 + 16);
        v9 = v14;
        v10 = *v8;
      }
      v12 = *(__int64 (**)())(v10 + 112);
      v13 = 0;
      if ( v12 != sub_1D00B10 )
        v13 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64 (*)(void), __int64, _QWORD))v12)(v8, a2, v11, a4, 0);
      LOBYTE(v7) = (unsigned int)sub_1E16810(v4, *(_DWORD *)(v9 + 112), 0, 1, v13) != -1;
    }
  }
  return v7;
}
