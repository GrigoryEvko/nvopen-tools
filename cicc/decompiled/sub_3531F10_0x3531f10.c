// Function: sub_3531F10
// Address: 0x3531f10
//
void __fastcall sub_3531F10(__int64 *a1, __int64 *a2, __int64 *a3, _BYTE *a4)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rdi

  v11 = *a3;
  v12 = *a2;
  v13 = *a1;
  if ( *a4 )
  {
    nullsub_2038();
  }
  else
  {
    v4 = v11;
    v5 = v13;
    if ( *(_QWORD *)(v13 + 112) == v11 )
    {
      v6 = v12;
      v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 16LL))(v12);
      v8 = *(_QWORD *)(v13 + 120);
      v9 = v4;
      *(_DWORD *)(v5 + 128) = v7;
      v10 = sub_2EAA2D0(v8, v4);
      if ( v10 )
      {
        v9 = v6;
        sub_3531C50(v5, v6, v10);
      }
      *(_DWORD *)(v5 + 132) = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v6 + 16LL))(v6, v9, v10);
    }
  }
}
