// Function: sub_920010
// Address: 0x920010
//
__int64 __fastcall sub_920010(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 i; // rbx
  __int64 v4; // rdi
  unsigned __int64 v5; // rcx
  unsigned int v6; // r8d
  __int64 j; // rax
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rcx
  int v10; // eax
  int v11; // edx
  int v12; // eax
  unsigned int v14; // eax
  unsigned __int64 v15; // [rsp+8h] [rbp-18h]

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v4 = *(_QWORD *)(a2 + 120);
  if ( *(char *)(v4 + 142) >= 0 && *(_BYTE *)(v4 + 140) == 12 )
  {
    v15 = a3;
    v14 = sub_8D4AB0(v4);
    a3 = v15;
    v5 = v14;
  }
  else
  {
    v5 = *(unsigned int *)(v4 + 136);
  }
  v6 = 0;
  if ( v5 <= a3 && !(a3 % v5) )
  {
    for ( j = *(_QWORD *)(a2 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v8 = *(_QWORD *)(a2 + 128);
    v9 = *(_QWORD *)(j + 128);
    v6 = 0;
    if ( v9 * (v8 / v9 + 1) <= *(_QWORD *)(i + 128) )
    {
      v10 = *(unsigned __int8 *)(a2 + 137) + *(unsigned __int8 *)(a2 + 136);
      v11 = v10 + 6;
      v12 = v10 - 1;
      if ( v12 < 0 )
        v12 = v11;
      LOBYTE(v6) = (v8 + (v12 >> 3)) / v9 == v8 / v9;
    }
  }
  return v6;
}
