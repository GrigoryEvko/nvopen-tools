// Function: sub_2E33000
// Address: 0x2e33000
//
__int64 __fastcall sub_2E33000(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // r12
  int v7; // ecx
  __int64 v8; // rdx
  __int64 v9; // rax

  v3 = (__int64 *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v3 == (__int64 *)(a1 + 48) )
    return 0;
  if ( !v3 )
    BUG();
  v4 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v5 = *v3;
  v6 = a2;
  v7 = *(_DWORD *)(v4 + 44);
  v8 = v7 & 0xFFFFFF;
  if ( (v5 & 4) != 0 )
  {
    if ( (v7 & 4) != 0 )
      goto LABEL_5;
  }
  else if ( (v7 & 4) != 0 )
  {
    while ( 1 )
    {
      v4 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      LOBYTE(v8) = *(_DWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 44);
      if ( (v8 & 4) == 0 )
        break;
      v5 = *(_QWORD *)v4;
    }
  }
  v8 &= 8u;
  if ( (_DWORD)v8 )
  {
    a2 = 32;
    LOBYTE(v9) = sub_2E88A90(v4, 32, 1);
    goto LABEL_6;
  }
LABEL_5:
  v9 = (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) >> 5) & 1LL;
LABEL_6:
  if ( !(_BYTE)v9 || !*(_DWORD *)(a1 + 120) )
    return 0;
  return (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v6 + 104LL))(v6, a2, v8);
}
