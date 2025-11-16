// Function: sub_1AE99B0
// Address: 0x1ae99b0
//
__int64 __fastcall sub_1AE99B0(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // ebx
  unsigned int v10; // ecx
  unsigned int v11; // r14d
  __int64 v13; // r15
  unsigned __int8 v14; // al
  unsigned int v15; // eax
  unsigned int v16; // ebx
  unsigned int v17; // eax
  __int64 v18; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  unsigned int v21; // [rsp+18h] [rbp-38h]

  sub_14C2530((__int64)&v18, a1, a3, 0, a5, a4, a6, 0);
  v7 = v19;
  if ( v19 > 0x40 )
  {
    LODWORD(_RAX) = sub_16A58F0((__int64)&v18);
  }
  else
  {
    _RDX = ~v18;
    __asm { tzcnt   rax, rdx }
    if ( v18 == -1 )
      LODWORD(_RAX) = 64;
  }
  v10 = v7 - 1;
  if ( v7 - 1 > 0x1F )
    v10 = 31;
  if ( v10 > (unsigned int)_RAX )
    LOBYTE(v10) = _RAX;
  v11 = 1 << v10;
  if ( (unsigned int)(1 << v10) > 0x20000000 )
    v11 = 0x20000000;
  if ( a2 > v11 )
  {
    v13 = sub_1649C60((__int64)a1);
    v14 = *(_BYTE *)(v13 + 16);
    if ( v14 > 0x17u )
    {
      if ( v14 == 53 )
      {
        v16 = (unsigned int)(1 << *(_WORD *)(v13 + 18)) >> 1;
        if ( v16 < v11 || (v11 = (unsigned int)(1 << *(_WORD *)(v13 + 18)) >> 1, a2 > v16) )
        {
          v17 = *(_DWORD *)(a3 + 8);
          if ( a2 <= v17 || !v17 )
          {
            v11 = a2;
            sub_15F8A20(v13, a2);
          }
        }
      }
    }
    else if ( v14 == 3 || !v14 )
    {
      v15 = (unsigned int)(1 << (*(_DWORD *)(v13 + 32) >> 15)) >> 1;
      if ( v15 < v11 || (v11 = (unsigned int)(1 << (*(_DWORD *)(v13 + 32) >> 15)) >> 1, a2 > v15) )
      {
        if ( sub_15E6530(v13) )
        {
          v11 = a2;
          sub_15E4CC0(v13, a2);
        }
      }
    }
  }
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return v11;
}
