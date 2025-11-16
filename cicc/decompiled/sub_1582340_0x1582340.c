// Function: sub_1582340
// Address: 0x1582340
//
__int64 __fastcall sub_1582340(__int64 *a1)
{
  __int64 v1; // rax
  __int64 *v2; // r12
  __int64 *v3; // rbx
  __int64 v4; // rdi
  unsigned int v5; // r13d

  v1 = 24LL * (*((_DWORD *)a1 + 5) & 0xFFFFFFF);
  if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
  {
    v2 = (__int64 *)*(a1 - 1);
    v3 = &v2[(unsigned __int64)v1 / 8];
  }
  else
  {
    v3 = a1;
    v2 = &a1[v1 / 0xFFFFFFFFFFFFFFF8LL];
  }
  while ( 1 )
  {
    v2 += 3;
    if ( v2 == v3 )
      break;
    v4 = *v2;
    if ( *(_BYTE *)(*v2 + 16) != 13 )
      return 0;
    v5 = *(_DWORD *)(v4 + 32);
    if ( v5 <= 0x40 )
    {
      if ( *(_QWORD *)(v4 + 24) )
        return 0;
    }
    else if ( v5 != (unsigned int)sub_16A57B0(v4 + 24) )
    {
      return 0;
    }
  }
  return 1;
}
