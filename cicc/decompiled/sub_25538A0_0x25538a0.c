// Function: sub_25538A0
// Address: 0x25538a0
//
__int64 __fastcall sub_25538A0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r13d
  __int64 v4; // rsi
  __int64 (__fastcall *v5)(__int64, unsigned int); // rax
  unsigned __int8 v6; // r14
  __int64 (__fastcall *v7)(__int64, char); // rdx
  _BOOL4 v8; // r13d
  char v9; // al
  unsigned int v10; // esi
  unsigned __int8 v11; // al

  v3 = *(_DWORD *)(a1 + 20);
  v4 = *(unsigned int *)(a2 + 20);
  v5 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)(a1 + 8) + 48LL);
  if ( v5 == sub_2535530 )
  {
    if ( v3 <= (unsigned int)v4 )
      LODWORD(v4) = *(_DWORD *)(a1 + 20);
    if ( *(_DWORD *)(a1 + 16) >= (unsigned int)v4 )
      LODWORD(v4) = *(_DWORD *)(a1 + 16);
    *(_DWORD *)(a1 + 20) = v4;
  }
  else
  {
    v5(a1 + 8, v4);
    LODWORD(v4) = *(_DWORD *)(a1 + 20);
  }
  v6 = *(_BYTE *)(a1 + 81);
  v7 = *(__int64 (__fastcall **)(__int64, char))(*(_QWORD *)(a1 + 72) + 48LL);
  v8 = v3 == v4;
  v9 = *(_BYTE *)(a2 + 81);
  if ( v7 == sub_25348C0 )
  {
    v10 = 1;
    if ( !v9 )
    {
      v11 = *(_BYTE *)(a1 + 80);
      *(_BYTE *)(a1 + 81) = v11;
      v10 = v6 == v11;
    }
    return sub_250C0B0(v8, v10);
  }
  else
  {
    v7(a1 + 72, v9);
    return sub_250C0B0(v8, *(_BYTE *)(a1 + 81) == v6);
  }
}
