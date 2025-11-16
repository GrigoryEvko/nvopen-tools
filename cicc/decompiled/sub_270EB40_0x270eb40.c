// Function: sub_270EB40
// Address: 0x270eb40
//
__int64 __fastcall sub_270EB40(unsigned __int8 *a1, __int64 a2)
{
  __int64 v2; // rcx
  unsigned __int8 *v3; // rbx
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  unsigned __int8 **v9; // rax
  char v10; // dl
  __int64 v11; // rax
  unsigned __int8 *v12; // r13

  v3 = sub_BD3990(a1, a2);
  v6 = *v3;
  if ( *v3 == 20 )
    return 1;
  v7 = (unsigned int)v6 - 12;
  if ( (unsigned int)v7 <= 1 )
    return 1;
  if ( v6 == 3 )
  {
    if ( (unsigned __int8)sub_A73380((__int64 *)v3 + 9, "objc_arc_inert", 0xEu) )
      return 1;
    v6 = *v3;
  }
  if ( v6 != 84 )
    return 0;
  if ( *(_BYTE *)(a2 + 28) )
  {
    v9 = *(unsigned __int8 ***)(a2 + 8);
    v2 = *(unsigned int *)(a2 + 20);
    v7 = (__int64)&v9[v2];
    if ( v9 != (unsigned __int8 **)v7 )
    {
      while ( v3 != *v9 )
      {
        if ( (unsigned __int8 **)v7 == ++v9 )
          goto LABEL_22;
      }
      return 1;
    }
LABEL_22:
    if ( (unsigned int)v2 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = v2 + 1;
      *(_QWORD *)v7 = v3;
      ++*(_QWORD *)a2;
      goto LABEL_16;
    }
  }
  sub_C8CC70(a2, (__int64)v3, v7, v2, v4, v5);
  if ( !v10 )
    return 1;
LABEL_16:
  v11 = 32LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF);
  if ( (v3[7] & 0x40) != 0 )
  {
    v12 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
    v3 = &v12[v11];
  }
  else
  {
    v12 = &v3[-v11];
  }
  if ( v12 == v3 )
    return 1;
  while ( (unsigned __int8)sub_270EB40(*(_QWORD *)v12, a2) )
  {
    v12 += 32;
    if ( v3 == v12 )
      return 1;
  }
  return 0;
}
