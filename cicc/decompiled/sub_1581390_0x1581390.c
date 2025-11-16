// Function: sub_1581390
// Address: 0x1581390
//
__int64 __fastcall sub_1581390(_BYTE *a1, unsigned int a2, unsigned int a3)
{
  _BYTE *v4; // r12
  unsigned int v5; // ebx
  char v6; // al
  unsigned int v7; // eax
  __int64 v8; // rax
  _QWORD *v9; // r15
  __int64 v11; // rsi
  unsigned int v13; // edx
  unsigned int v14; // eax
  unsigned int v15; // edx
  unsigned int v16; // edi
  __int64 v17; // rax
  unsigned int v18; // ebx
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // r8
  _QWORD *v24; // rax
  unsigned int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rdi
  _QWORD *v28; // rdx
  unsigned int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned __int64 v35; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v36; // [rsp+8h] [rbp-48h]
  unsigned __int64 v37; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v38; // [rsp+18h] [rbp-38h]

  v4 = a1;
  v5 = 8 * a3;
  v6 = a1[16];
  if ( v6 == 13 )
  {
LABEL_2:
    v7 = *((_DWORD *)v4 + 8);
    v36 = v7;
    if ( v7 <= 0x40 )
    {
      v35 = *((_QWORD *)v4 + 3);
      if ( !a2 )
        goto LABEL_4;
      LODWORD(v11) = 8 * a2;
LABEL_19:
      if ( (_DWORD)v11 == v7 )
        v35 = 0;
      else
        v35 >>= v11;
      goto LABEL_4;
    }
    sub_16A4FD0(&v35, v4 + 24);
    if ( a2 )
    {
      v7 = v36;
      v11 = 8 * a2;
      if ( v36 <= 0x40 )
        goto LABEL_19;
      sub_16A8110(&v35, v11);
    }
LABEL_4:
    sub_16A5A50(&v37, &v35);
    if ( v36 > 0x40 && v35 )
      j_j___libc_free_0_0(v35);
    v35 = v37;
    v36 = v38;
    v8 = sub_16498A0(v4);
    v9 = (_QWORD *)sub_159C0E0(v8, &v35);
    if ( v36 > 0x40 )
    {
      if ( v35 )
        j_j___libc_free_0_0(v35);
    }
    return (__int64)v9;
  }
  v13 = *(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 11;
  while ( 2 )
  {
    if ( v6 != 5 )
      return 0;
    switch ( *((_WORD *)v4 + 9) )
    {
      case 0x17:
        v26 = *((_DWORD *)v4 + 5) & 0xFFFFFFF;
        v27 = *(_QWORD *)&v4[24 * (1 - v26)];
        if ( *(_BYTE *)(v27 + 16) != 13 )
          return 0;
        v28 = *(_QWORD **)(v27 + 24);
        if ( *(_DWORD *)(v27 + 32) > 0x40u )
          v28 = (_QWORD *)*v28;
        if ( ((unsigned __int8)v28 & 7) != 0 )
          return 0;
        v29 = (unsigned int)v28 >> 3;
        if ( a3 + a2 <= v29 )
          goto LABEL_56;
        if ( v29 > a2 )
          return 0;
        a2 -= v29;
        v4 = *(_BYTE **)&v4[-24 * v26];
        v15 = *(_DWORD *)(*(_QWORD *)v4 + 8LL) >> 8;
        goto LABEL_27;
      case 0x18:
        v22 = *((_DWORD *)v4 + 5) & 0xFFFFFFF;
        v23 = *(_QWORD *)&v4[24 * (1 - v22)];
        if ( *(_BYTE *)(v23 + 16) != 13 )
          return 0;
        v24 = *(_QWORD **)(v23 + 24);
        if ( *(_DWORD *)(v23 + 32) > 0x40u )
          v24 = (_QWORD *)*v24;
        if ( ((unsigned __int8)v24 & 7) != 0 )
          return 0;
        v25 = (unsigned int)v24 >> 3;
        if ( v13 - v25 <= a2 )
          goto LABEL_56;
        if ( v25 + a3 + a2 > v13 )
          return 0;
        a2 += v25;
        v4 = *(_BYTE **)&v4[-24 * v22];
        v15 = *(_DWORD *)(*(_QWORD *)v4 + 8LL) >> 8;
        goto LABEL_27;
      case 0x1A:
        v20 = sub_1581390(*(_QWORD *)&v4[24 * (1LL - (*((_DWORD *)v4 + 5) & 0xFFFFFFF))], a2, a3);
        v9 = (_QWORD *)v20;
        if ( !v20 )
          return 0;
        if ( (unsigned __int8)sub_1593BB0(v20) )
          return (__int64)v9;
        v21 = sub_1581390(*(_QWORD *)&v4[-24 * (*((_DWORD *)v4 + 5) & 0xFFFFFFF)], a2, a3);
        if ( !v21 )
          return 0;
        return sub_15A2CF0(v21, v9);
      case 0x1B:
        v17 = sub_1581390(*(_QWORD *)&v4[24 * (1LL - (*((_DWORD *)v4 + 5) & 0xFFFFFFF))], a2, a3);
        v9 = (_QWORD *)v17;
        if ( !v17 )
          return 0;
        if ( *(_BYTE *)(v17 + 16) != 13 )
          goto LABEL_33;
        v18 = *(_DWORD *)(v17 + 32);
        if ( v18 <= 0x40 )
        {
          if ( *(_QWORD *)(v17 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18) )
            return (__int64)v9;
        }
        else if ( v18 == (unsigned int)sub_16A58F0(v17 + 24) )
        {
          return (__int64)v9;
        }
LABEL_33:
        v19 = sub_1581390(*(_QWORD *)&v4[-24 * (*((_DWORD *)v4 + 5) & 0xFFFFFFF)], a2, a3);
        if ( !v19 )
          return 0;
        return sub_15A2D10(v19, v9);
      case 0x25:
        v9 = *(_QWORD **)&v4[-24 * (*((_DWORD *)v4 + 5) & 0xFFFFFFF)];
        v14 = *(_DWORD *)(*v9 + 8LL) >> 8;
        v15 = v14;
        if ( 8 * a2 >= v14 )
        {
LABEL_56:
          v30 = sub_16498A0(v4);
          v31 = sub_1644900(v30, v5);
          return sub_15A06D0(v31);
        }
        if ( !a2 && v14 == v5 )
          return (__int64)v9;
        v16 = 8 * (a3 + a2);
        if ( (v14 & 7) == 0 )
        {
          if ( v14 < v16 )
            return 0;
          v4 = *(_BYTE **)&v4[-24 * (*((_DWORD *)v4 + 5) & 0xFFFFFFF)];
LABEL_27:
          v6 = v4[16];
          v13 = v15 >> 3;
          if ( v6 == 13 )
            goto LABEL_2;
          continue;
        }
        if ( v14 <= v16 )
          return 0;
        if ( a2 )
        {
          v34 = sub_15A0680(*v9, 8 * a2, 0);
          v9 = (_QWORD *)sub_15A2D80(v9, v34, 0);
        }
        v32 = sub_16498A0(v4);
        v33 = sub_1644900(v32, v5);
        return sub_15A43B0(v9, v33, 0);
      default:
        return 0;
    }
  }
}
