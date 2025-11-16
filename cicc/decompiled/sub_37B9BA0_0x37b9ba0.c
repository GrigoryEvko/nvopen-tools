// Function: sub_37B9BA0
// Address: 0x37b9ba0
//
__int64 __fastcall sub_37B9BA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // eax
  __int64 v5; // rax
  _DWORD *v6; // rax

  v2 = *(_QWORD *)(a2 + 48);
  v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  if ( (v2 & 7) == 0 )
  {
    *(_QWORD *)(a2 + 48) = v3;
    LOBYTE(v2) = v2 & 0xF8;
    goto LABEL_4;
  }
  if ( (v2 & 7) != 3 || *(_DWORD *)v3 != 1 )
    return 0;
LABEL_4:
  v4 = v2 & 7;
  if ( v4 )
  {
    if ( v4 != 3 )
      BUG();
    v3 = *(_QWORD *)(v3 + 16);
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v3;
  }
  if ( (*(_BYTE *)(v3 + 32) & 2) == 0 )
    return 0;
  v5 = *(_QWORD *)v3;
  if ( !*(_QWORD *)v3 )
    return 0;
  if ( (v5 & 4) == 0 )
    return 0;
  v6 = (_DWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v6 || v6[2] != 4 )
    return 0;
  return (*(unsigned int (__fastcall **)(_DWORD *, _QWORD))(*(_QWORD *)v6 + 32LL))(v6, *(_QWORD *)(a1 + 48)) ^ 1;
}
