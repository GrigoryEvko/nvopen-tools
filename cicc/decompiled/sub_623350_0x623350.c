// Function: sub_623350
// Address: 0x623350
//
__int64 __fastcall sub_623350(__int64 a1, unsigned int *a2, __int64 a3)
{
  unsigned int *v4; // r12
  __int64 v5; // rbx
  _BOOL4 v6; // r14d
  __int64 v7; // rax
  char v8; // cl
  __int64 v9; // rdx
  int v10; // r15d
  __int64 v11; // rdi
  __int64 v12; // rdx
  _BOOL8 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 result; // rax
  __int64 v18; // rax

  v4 = a2;
  v5 = a1;
  v6 = (*(_BYTE *)(a1 + 122) & 8) != 0;
  v7 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v8 = *(_BYTE *)(v7 + 4);
  if ( v8 == 8 )
  {
    v10 = 0;
    if ( (*(_BYTE *)(a1 + 8) & 8) != 0 )
      goto LABEL_4;
    a1 = 1;
    v9 = 776LL * *(int *)(v7 + 552) + qword_4F04C68[0];
    a2 = (unsigned int *)*(unsigned __int8 *)(v9 + 4);
  }
  else
  {
    a2 = (unsigned int *)*(unsigned __int8 *)(v7 + 4);
    v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    a1 = 0;
  }
  v10 = 0;
  if ( (_BYTE)a2 == 6 )
  {
    a2 = *(unsigned int **)(v9 + 208);
    if ( (unsigned __int8)(*((_BYTE *)a2 + 140) - 9) <= 2u && (a2[44] & 0x11000) == 0x1000 && *((char *)a2 + 177) >= 0 )
    {
      a1 = (unsigned int)a1 ^ 1;
      v6 = 1;
      v10 = a1;
    }
  }
LABEL_4:
  if ( *(_BYTE *)(*(_QWORD *)(v5 + 288) + 140LL) != 7
    || (dword_4F04C44 == -1 && (*(_BYTE *)(v7 + 6) & 6) == 0 && v8 != 12 || (*(_BYTE *)(v5 + 125) & 0x40) == 0) && !v6 )
  {
    a2 = &dword_4F063F8;
    a1 = 3049;
    sub_6851C0(3049, &dword_4F063F8);
  }
  if ( !a3 )
  {
    sub_865B10(a1);
    goto LABEL_13;
  }
  v11 = *(_QWORD *)(a3 + 32);
  if ( (*(_BYTE *)(a3 + 18) & 2) == 0 )
  {
    if ( v11 )
    {
      a2 = 0;
      sub_864360(v11, 0);
    }
LABEL_13:
    v12 = *(_QWORD *)(v5 + 288);
    if ( *(_BYTE *)(v12 + 140) != 7 )
      goto LABEL_14;
    goto LABEL_23;
  }
  if ( (unsigned __int8)(*(_BYTE *)(v11 + 140) - 9) > 2u )
    goto LABEL_13;
  a2 = 0;
  sub_8646E0(v11, 0);
  v12 = *(_QWORD *)(v5 + 288);
  if ( *(_BYTE *)(v12 + 140) != 7 )
    goto LABEL_14;
LABEL_23:
  if ( !v4 )
  {
LABEL_14:
    v13 = v6;
    *(_QWORD *)(v5 + 400) = sub_6DE780(v6);
    if ( a3 )
      goto LABEL_15;
LABEL_25:
    result = sub_8640D0();
    goto LABEL_18;
  }
  a2 = (unsigned int *)v4[10];
  sub_8600D0(1, a2, v12, 0);
  v18 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_BYTE *)(v18 + 11) |= 0x40u;
  *(_QWORD *)(v18 + 624) = v5;
  sub_886000(*(_QWORD *)v4);
  v13 = v6;
  *(_QWORD *)(v5 + 400) = sub_6DE780(v6);
  sub_863FC0();
  if ( !a3 )
    goto LABEL_25;
LABEL_15:
  result = *(_QWORD *)(a3 + 32);
  if ( (*(_BYTE *)(a3 + 18) & 2) != 0 )
  {
    result = (unsigned int)*(unsigned __int8 *)(result + 140) - 9;
    if ( (unsigned __int8)result <= 2u )
      result = sub_866010(v13, a2, v14, v15, v16);
  }
  else if ( result )
  {
    result = sub_8645D0();
  }
LABEL_18:
  if ( v10 )
    *(_BYTE *)(v5 + 133) |= 2u;
  return result;
}
