// Function: sub_8156E0
// Address: 0x8156e0
//
void __fastcall sub_8156E0(__int64 a1, unsigned __int8 a2, __int64 a3, int a4, _QWORD *a5)
{
  int v5; // r15d
  char *v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // r14
  _QWORD *v13; // rdi
  __int64 v14; // rax
  char v15; // si
  char *v16; // rcx
  __int64 v17; // rdx
  char v18; // r11
  unsigned int v19; // r8d
  char *v20; // rdi
  unsigned int v21; // esi
  _QWORD *v22; // rdi
  __int64 v23; // rax
  char *v24; // rdi
  int v25; // [rsp+14h] [rbp-5Ch] BYREF
  __int64 *v26; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v27[2]; // [rsp+20h] [rbp-50h] BYREF
  char v28; // [rsp+30h] [rbp-40h]

  v5 = a4;
  v26 = 0;
  if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
  {
LABEL_2:
    if ( v5 )
    {
      *a5 += 2LL;
      sub_8238B0(qword_4F18BE0, "ad", 2);
      if ( a3 )
      {
LABEL_4:
        v9 = *(char **)(a3 + 32);
        v10 = *(_QWORD *)(a3 + 24);
        if ( v9 )
        {
          sub_812220(*(_BYTE *)(a3 + 16), 0, *(_QWORD *)(a3 + 8), v9, v10, 0, 1, a5);
          return;
        }
        if ( v10 )
        {
          sub_812380(a1, *(_QWORD *)(a3 + 24), 0, a5);
          return;
        }
      }
    }
    else if ( a3 )
    {
      goto LABEL_4;
    }
    *a5 += 3LL;
    sub_8238B0(qword_4F18BE0, &unk_3C1BC3F, 3);
    v19 = 0;
    if ( a2 == 11 )
    {
      if ( *(_BYTE *)(a1 + 172) == 2 )
        v19 = ((*(_BYTE *)(a1 + 89) >> 2) ^ 1) & 1;
      v21 = (*(_BYTE *)(a1 + 88) & 0x70) == 48;
      if ( dword_4D0425C && !*(_BYTE *)(a1 + 174) && *(_WORD *)(a1 + 176) )
        v21 = 1;
      sub_8111C0(a1, v21, 0, 1, v19, 0, (__int64)a5);
    }
    else
    {
      if ( a2 == 7 )
        v19 = *(_BYTE *)(a1 + 136) == 2;
      v25 = 0;
      sub_811730(a1, a2, &v25, (__int64 *)&v26, v19, (__int64)a5);
      if ( a3 )
      {
        sub_810E90(a1, *(_BYTE *)a3, *(_BYTE *)(a3 + 16), 0, 0, *(_QWORD *)(a3 + 8), *(char **)(a3 + 32), (__int64)a5);
        if ( *(_QWORD *)(a3 + 24) )
        {
          v27[0] = *(_QWORD *)(a3 + 24);
          sub_811CB0(v27, 0, 0, a5);
        }
      }
      else
      {
        if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
          v24 = *(char **)(a1 + 24);
        else
          v24 = *(char **)(a1 + 8);
        sub_80BC40(v24, a5);
      }
      sub_80C110(v25, v26, a5);
    }
    v22 = (_QWORD *)qword_4F18BE0;
    ++*a5;
    v23 = v22[2];
    if ( (unsigned __int64)(v23 + 1) > v22[1] )
    {
      sub_823810(v22);
      v22 = (_QWORD *)qword_4F18BE0;
      v23 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v22[4] + v23) = 69;
    ++v22[2];
    return;
  }
  v11 = *(_QWORD *)(a1 + 40);
  v12 = *(_QWORD *)(v11 + 32);
  if ( dword_4D0425C && unk_4D04250 <= 0x76BFu )
  {
    if ( !v12 )
      goto LABEL_2;
    LOBYTE(v5) = 1;
    if ( !a4 )
      goto LABEL_29;
    goto LABEL_45;
  }
  if ( !v12 || !(unsigned int)sub_8DBE70(*(_QWORD *)(v11 + 32)) )
    goto LABEL_2;
  if ( v5 )
  {
    LOBYTE(v5) = 0;
LABEL_45:
    *a5 += 2LL;
    sub_8238B0(qword_4F18BE0, "ad", 2);
  }
  if ( !dword_4D0425C )
  {
    v25 = 0;
    v27[0] = 0;
    v27[1] = v12;
    v28 = 6;
    sub_812470((unsigned int *)v27, &v25, 1u, a5);
    if ( v25 )
    {
      v13 = (_QWORD *)qword_4F18BE0;
      ++*a5;
      v14 = v13[2];
      if ( (unsigned __int64)(v14 + 1) > v13[1] )
      {
        sub_823810(v13);
        v13 = (_QWORD *)qword_4F18BE0;
        v14 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(v13[4] + v14) = 69;
      ++v13[2];
    }
    if ( a2 == 11 )
      goto LABEL_30;
    goto LABEL_18;
  }
LABEL_29:
  *a5 += 2LL;
  sub_8238B0(qword_4F18BE0, "sr", 2);
  sub_80F5E0(v12, 0, a5);
  if ( a2 == 11 )
  {
LABEL_30:
    sub_8111C0(a1, (unsigned __int8)v5 ^ 1, (unsigned __int8)v5 ^ 1, 1, 0, 0, (__int64)a5);
    return;
  }
LABEL_18:
  v25 = 0;
  if ( (_BYTE)v5 && !(unsigned int)sub_8DBE70(v12) )
    sub_811730(a1, a2, &v25, (__int64 *)&v26, 0, (__int64)a5);
  if ( a3 )
  {
    v15 = *(_BYTE *)a3;
    v16 = *(char **)(a3 + 32);
    v17 = *(_QWORD *)(a3 + 8);
    v18 = *(_BYTE *)(a3 + 16);
    if ( !dword_4D0425C && v15 )
      sub_812220(v18, 0, v17, v16, *(_QWORD *)(a3 + 24), 0, 0, a5);
    else
      sub_810E90(a1, v15, v18, 0, 0, v17, v16, (__int64)a5);
    if ( *(_QWORD *)(a3 + 24) )
    {
      v27[0] = *(_QWORD *)(a3 + 24);
      sub_811CB0(v27, 0, 0, a5);
    }
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
      v20 = *(char **)(a1 + 24);
    else
      v20 = *(char **)(a1 + 8);
    sub_80BC40(v20, a5);
  }
  sub_80C110(v25, v26, a5);
}
