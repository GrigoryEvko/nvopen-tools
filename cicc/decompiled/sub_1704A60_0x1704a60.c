// Function: sub_1704A60
// Address: 0x1704a60
//
bool __fastcall sub_1704A60(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 *v3; // r12
  __int64 *v4; // rbx
  __int64 *i; // r12
  __int64 v6; // rdi
  unsigned int v7; // r14d
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 *v11; // r12
  __int64 *v12; // rbx
  __int64 v13; // rdi
  unsigned int v14; // r14d
  bool result; // al
  __int64 v16; // rax

  v2 = 24LL * (*((_DWORD *)a1 + 5) & 0xFFFFFFF);
  if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
  {
    v3 = (__int64 *)*(a1 - 1);
    v4 = &v3[(unsigned __int64)v2 / 8];
  }
  else
  {
    v4 = a1;
    v3 = &a1[v2 / 0xFFFFFFFFFFFFFFF8LL];
  }
  for ( i = v3 + 3; v4 != i; i += 3 )
  {
    v6 = *i;
    if ( *(_BYTE *)(*i + 16) != 13 )
      return 1;
    v7 = *(_DWORD *)(v6 + 32);
    if ( !(v7 <= 0x40 ? *(_QWORD *)(v6 + 24) == 0 : v7 == (unsigned int)sub_16A57B0(v6 + 24)) )
      return 1;
  }
  v9 = 24LL * (*((_DWORD *)a2 + 5) & 0xFFFFFFF);
  if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
  {
    v10 = (__int64 *)*(a2 - 1);
    v11 = &v10[(unsigned __int64)v9 / 8];
  }
  else
  {
    v11 = a2;
    v10 = &a2[v9 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v12 = v10 + 3;
  if ( v11 == v12 )
    return 1;
  while ( 1 )
  {
    v13 = *v12;
    if ( *(_BYTE *)(*v12 + 16) != 13 )
      break;
    v14 = *(_DWORD *)(v13 + 32);
    result = v14 <= 0x40 ? *(_QWORD *)(v13 + 24) == 0 : v14 == (unsigned int)sub_16A57B0(v13 + 24);
    if ( !result )
      break;
    v12 += 3;
    if ( v11 == v12 )
      return result;
  }
  v16 = a2[1];
  return v16 && *(_QWORD *)(v16 + 8) == 0;
}
