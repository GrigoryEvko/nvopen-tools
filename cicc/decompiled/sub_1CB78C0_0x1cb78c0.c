// Function: sub_1CB78C0
// Address: 0x1cb78c0
//
__int64 __fastcall sub_1CB78C0(unsigned int *a1, unsigned __int64 a2)
{
  unsigned int v2; // ebx
  unsigned __int64 *v3; // rax
  unsigned int v4; // r12d
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned int v8; // eax
  int v9; // eax
  int v10; // r8d
  __int64 result; // rax

  v2 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v3 = *(unsigned __int64 **)(a2 - 8);
  else
    v3 = (unsigned __int64 *)(a2 - 24LL * v2);
  v4 = sub_1CB76C0(a1, *v3);
  if ( v2 > 1 )
  {
    v5 = 24;
    v6 = 24LL * (v2 - 2) + 48;
    while ( 1 )
    {
      v7 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v8 = sub_1CB76C0(a1, *(_QWORD *)(v7 + v5));
      v9 = sub_1CB71C0((__int64)a1, v8, v4);
      v4 = v9;
      if ( a1[1] == v9 )
        break;
      v5 += 24;
      if ( v5 == v6 )
        goto LABEL_11;
    }
    sub_1CB7560(a1, a2, v9);
  }
LABEL_11:
  v10 = sub_1CB76C0(a1, a2);
  result = 0;
  if ( v10 != v4 )
  {
    sub_1CB7560(a1, a2, v4);
    return 1;
  }
  return result;
}
