// Function: sub_C35EF0
// Address: 0xc35ef0
//
__int64 __fastcall sub_C35EF0(__int64 a1)
{
  __int64 v2; // r13
  signed int v3; // ebx
  __int64 v4; // r14
  int v5; // r15d
  unsigned int v6; // r8d
  __int64 v7; // rax
  int v10; // [rsp-44h] [rbp-44h]
  __int64 v11; // [rsp-40h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 20) & 7) == 3 || (*(_BYTE *)(a1 + 20) & 6) == 0 )
    return 0x80000000LL;
  v2 = sub_C33930(a1);
  v11 = *(_QWORD *)a1;
  v10 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
  v3 = (unsigned int)(v10 + 63) >> 6;
  if ( !v3 )
    v3 = 1;
  v4 = 0;
  v5 = 0;
  do
  {
    v5 += sub_39FAC40(*(_QWORD *)(v2 + 8 * v4));
    if ( v5 > 1 )
      return 0x80000000;
    ++v4;
  }
  while ( v3 > (int)v4 );
  v6 = *(_DWORD *)(a1 + 16);
  if ( v6 == *(_DWORD *)(v11 + 4) )
  {
    v7 = 0;
    while ( !*(_QWORD *)(v2 + 8 * v7) )
    {
      if ( v3 <= (int)++v7 )
        BUG();
    }
    __asm { tzcnt   rdx, rdx }
    v6 += _RDX + ((_DWORD)v7 << 6) - v10 + 1;
  }
  return v6;
}
