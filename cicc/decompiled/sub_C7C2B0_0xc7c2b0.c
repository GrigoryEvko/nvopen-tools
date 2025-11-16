// Function: sub_C7C2B0
// Address: 0xc7c2b0
//
__int64 __fastcall sub_C7C2B0(__int64 a1, __int64 a2)
{
  unsigned int v3; // ebx
  unsigned int v4; // r15d
  unsigned int v5; // esi
  unsigned int v6; // esi
  unsigned int v9; // edx
  char v10; // cl
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned int v15; // eax

  v3 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v3;
  if ( v3 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = v3;
    sub_C43690(a1 + 16, 0, 0);
    v4 = *(_DWORD *)(a1 + 8);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    v4 = v3;
    *(_DWORD *)(a1 + 24) = v3;
    *(_QWORD *)(a1 + 16) = 0;
  }
  v5 = *(_DWORD *)(a2 + 24);
  if ( v5 <= 0x40 )
  {
    _RDX = *(_QWORD *)(a2 + 16);
    v15 = 64;
    __asm { tzcnt   rcx, rdx }
    if ( _RDX )
      v15 = _RCX;
    if ( v5 > v15 )
      v5 = v15;
  }
  else
  {
    v5 = sub_C44590(a2 + 16);
  }
  v6 = v5 + 1;
  if ( v6 > v3 )
    v6 = v3;
  if ( v6 != v4 )
  {
    if ( v6 > 0x3F || v4 > 0x40 )
      sub_C43C90((_QWORD *)a1, v6, v4);
    else
      *(_QWORD *)a1 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v6 - (unsigned __int8)v4 + 64) << v6;
  }
  if ( *(_DWORD *)(a2 + 8) > 0x40u )
  {
    v9 = sub_C445E0(a2) + 1;
  }
  else
  {
    _RDX = ~*(_QWORD *)a2;
    if ( *(_QWORD *)a2 != -1 )
    {
      __asm { tzcnt   rdx, rdx }
      v9 = _RDX + 1;
      if ( v9 > v3 )
        v9 = v3;
      if ( v9 )
      {
LABEL_16:
        v10 = 64 - v9;
        v11 = *(_QWORD *)(a1 + 16);
        v12 = 0xFFFFFFFFFFFFFFFFLL >> v10;
        if ( *(_DWORD *)(a1 + 24) > 0x40u )
          *(_QWORD *)v11 |= v12;
        else
          *(_QWORD *)(a1 + 16) = v11 | v12;
        return a1;
      }
      return a1;
    }
    v9 = 65;
  }
  if ( v3 <= v9 )
    v9 = v3;
  if ( v9 )
  {
    if ( v9 > 0x40 )
    {
      sub_C43C90((_QWORD *)(a1 + 16), 0, v9);
      return a1;
    }
    goto LABEL_16;
  }
  return a1;
}
