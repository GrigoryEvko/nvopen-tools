// Function: sub_2A66C70
// Address: 0x2a66c70
//
_BOOL8 __fastcall sub_2A66C70(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned __int8 v5; // cl
  _BOOL4 v6; // r15d
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rsi

  v3 = sub_2A66C60(a1, a2);
  if ( !v3 )
    return 0;
  v4 = v3;
  if ( *(_BYTE *)a2 <= 0x1Cu )
    goto LABEL_20;
  v5 = *(_BYTE *)a2 - 34;
  if ( v5 > 0x33u )
    goto LABEL_20;
  v6 = ((0x8000000000041uLL >> v5) & 1) == 0;
  if ( ((0x8000000000041uLL >> v5) & 1) == 0 )
    goto LABEL_20;
  if ( sub_B49200(a2) && !sub_F509B0((unsigned __int8 *)a2, 0) )
    goto LABEL_16;
  if ( *(char *)(a2 + 7) >= 0 )
    goto LABEL_20;
  v11 = sub_BD2BC0(a2);
  v12 = (__int64)v7 + v11;
  if ( *(char *)(a2 + 7) < 0 )
    v12 -= sub_BD2BC0(a2);
  v13 = v12 >> 4;
  if ( !(_DWORD)v13 )
  {
LABEL_20:
    v6 = 1;
    sub_BD84D0(a2, v4);
    return v6;
  }
  v14 = 0;
  v15 = 16LL * (unsigned int)v13;
  while ( 1 )
  {
    v16 = 0;
    if ( *(char *)(a2 + 7) < 0 )
      v16 = sub_BD2BC0(a2);
    if ( *(_DWORD *)(*(_QWORD *)(v16 + v14) + 8LL) == 6 )
      break;
    v14 += 16;
    if ( v14 == v15 )
      goto LABEL_20;
  }
LABEL_16:
  v17 = *(_QWORD *)(a2 - 32);
  if ( !v17 || *(_BYTE *)v17 || *(_QWORD *)(v17 + 24) != *(_QWORD *)(a2 + 80) )
    return 0;
  sub_2A64050(a1, v17, v7, v8, v9, v10);
  return v6;
}
