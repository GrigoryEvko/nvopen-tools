// Function: sub_DFFEB0
// Address: 0xdffeb0
//
__int64 __fastcall sub_DFFEB0(__int64 a1)
{
  unsigned int v1; // eax
  __int64 v2; // rcx
  unsigned int v3; // r12d
  unsigned __int8 v4; // al
  __int64 v6; // rax
  _BYTE *v7; // rsi
  bool v8; // zf
  unsigned __int8 v9; // al
  __int64 v10; // rax
  _BYTE *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  _BOOL4 v14; // eax
  __int64 *v15; // rax
  __int64 v16; // rdx

  LOBYTE(v1) = sub_DFF600(a1);
  v3 = v1;
  v4 = *(_BYTE *)(v2 - 16);
  if ( (_BYTE)v3 )
  {
    if ( (*(_BYTE *)(v2 - 16) & 2) != 0 )
      v6 = *(_QWORD *)(v2 - 32);
    else
      v6 = v2 - 8LL * ((v4 >> 2) & 0xF) - 16;
    v7 = *(_BYTE **)(v6 + 8);
    if ( v7 && (unsigned __int8)(*v7 - 5) >= 0x20u )
      v7 = 0;
    v8 = !sub_DFF670((__int64)v7);
    v9 = *(v7 - 16);
    if ( (v9 & 2) != 0 )
      v10 = *((_QWORD *)v7 - 4);
    else
      v10 = (__int64)&v7[-8 * ((v9 >> 2) & 0xF) - 16];
    v11 = *(_BYTE **)(v10 + 16LL * !v8);
    if ( *v11 )
      return 0;
    v12 = sub_B91420((__int64)v11);
    if ( v13 != 14 )
      return 0;
  }
  else
  {
    if ( (*(_BYTE *)(v2 - 16) & 2) != 0 )
    {
      if ( !*(_DWORD *)(v2 - 24) )
        return 0;
      v15 = *(__int64 **)(v2 - 32);
    }
    else
    {
      if ( (*(_WORD *)(v2 - 16) & 0x3C0) == 0 )
        return 0;
      v15 = (__int64 *)(v2 - 8LL * ((v4 >> 2) & 0xF) - 16);
    }
    if ( *(_BYTE *)*v15 )
      return 0;
    v12 = sub_B91420(*v15);
    if ( v16 != 14 )
      return v3;
  }
  v14 = *(_QWORD *)v12 != 0x7020656C62617476LL || *(_DWORD *)(v12 + 8) != 1953393007 || *(_WORD *)(v12 + 12) != 29285;
  LOBYTE(v3) = !v14;
  return v3;
}
