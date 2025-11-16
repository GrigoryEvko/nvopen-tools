// Function: sub_8D62B0
// Address: 0x8d62b0
//
__int64 __fastcall sub_8D62B0(__int64 a1, int a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  char v5; // al
  int v6; // r13d
  unsigned __int64 v7; // r15
  __int64 i; // r12
  char v9; // al
  __int64 v10; // rdi
  char v11; // al
  unsigned __int64 v12; // rcx

  v3 = sub_8D40F0(a1);
  v4 = v3;
  if ( *(char *)(v3 + 142) < 0 )
  {
    v6 = *(_DWORD *)(v3 + 136);
  }
  else
  {
    v5 = *(_BYTE *)(v3 + 140);
    if ( v5 != 12 )
    {
      v6 = *(_DWORD *)(v4 + 136);
      goto LABEL_4;
    }
    v6 = sub_8D4AB0(v4);
  }
  while ( 1 )
  {
    v5 = *(_BYTE *)(v4 + 140);
    if ( v5 != 12 )
      break;
    v4 = *(_QWORD *)(v4 + 160);
  }
LABEL_4:
  if ( (*(_BYTE *)(v4 + 141) & 0x20) != 0
    && ((unsigned __int8)(v5 - 9) <= 2u || v5 == 2 && (*(_BYTE *)(v4 + 161) & 8) != 0) )
  {
    sub_880320((__int64 *)v4, 2, a1, 6, (__int64 *)dword_4F07508);
    *(_BYTE *)(a1 + 141) |= 0x20u;
    *(_QWORD *)(a1 + 128) = 0;
    *(_DWORD *)(a1 + 136) = 1;
    return 1;
  }
  v7 = 1;
  if ( (*(_WORD *)(a1 + 168) & 0x180) == 0 )
    v7 = *(_QWORD *)(a1 + 176);
  for ( i = *(_QWORD *)(a1 + 160); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( sub_8D3410(i) )
    sub_8D6090(i);
  v9 = 1;
  if ( (*(_BYTE *)(i + 141) & 0x20) == 0 )
  {
    v9 = 0;
    if ( !v7 )
      v9 = ((*(_BYTE *)(a1 + 169) >> 5) ^ 1) & 1;
  }
  v10 = *(_QWORD *)(a1 + 160);
  *(_BYTE *)(a1 + 141) = *(_BYTE *)(a1 + 141) & 0xDF | (32 * v9);
  v11 = *(_BYTE *)(v10 + 140);
  if ( v11 == 12 )
  {
    v12 = sub_8D4A00(v10);
  }
  else
  {
    if ( dword_4F077C0 && (v11 == 1 || v11 == 7) )
    {
      v12 = 1;
      goto LABEL_22;
    }
    v12 = *(_QWORD *)(v10 + 128);
  }
  if ( !v12 )
  {
LABEL_24:
    *(_DWORD *)(a1 + 136) = v6;
    *(_QWORD *)(a1 + 128) = v12;
    return 1;
  }
LABEL_22:
  if ( unk_4F06A58 / v12 >= v7 )
  {
    v12 *= v7;
    goto LABEL_24;
  }
  if ( !a2 )
    sub_6851C0(0x5Fu, dword_4F07508);
  sub_725570(a1, 0);
  sub_8D6090(a1);
  return 0;
}
