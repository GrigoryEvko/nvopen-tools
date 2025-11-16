// Function: sub_3531E60
// Address: 0x3531e60
//
void __fastcall sub_3531E60(__int64 *a1, __int64 *a2, __int64 *a3, _BYTE *a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rcx
  int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdi

  v14 = *a3;
  v15 = *a2;
  v16 = *a1;
  if ( !*a4 || (v4 = *(_QWORD *)(v14 + 32), v5 = v14 + 24, v4 == v5) )
  {
    nullsub_2037();
  }
  else
  {
    v6 = v15;
    v7 = v16;
    v8 = *(_QWORD *)(v16 + 112);
    while ( 1 )
    {
      v9 = v4 - 56;
      if ( !v4 )
        v9 = 0;
      if ( v8 == v9 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v5 == v4 )
        return;
    }
    v10 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v15 + 16LL))(v15, 0);
    v11 = *(_QWORD *)(v16 + 120);
    v12 = v8;
    *(_DWORD *)(v7 + 128) = v10;
    v13 = sub_2EAA2D0(v11, v8);
    if ( v13 )
    {
      v12 = v6;
      sub_3531C50(v7, v6, v13);
    }
    *(_DWORD *)(v7 + 132) = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v6 + 16LL))(v6, v12, v13);
  }
}
