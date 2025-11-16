// Function: sub_6935B0
// Address: 0x6935b0
//
__int64 __fastcall sub_6935B0(__int64 a1, int a2, _QWORD *a3)
{
  int v3; // r8d
  _QWORD *v4; // r10
  int v5; // r11d
  __int64 v6; // r9
  __int64 v7; // rcx
  char v8; // dl
  __int64 v9; // rax
  __int64 v10; // rax
  int i; // ecx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 j; // rdx
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rcx
  __int64 v21; // rax
  __int64 v22; // r11
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // esi
  unsigned int v26; // edx
  _QWORD *v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // r9
  unsigned int v30; // edi
  __int64 *v31; // rax

  v3 = a2;
  v4 = a3;
  if ( a3 )
    *a3 = 0;
  v5 = dword_4F04C64;
  v6 = qword_4F04C68[0];
  if ( a2 != -1 )
    goto LABEL_48;
  if ( sub_6879B0() )
  {
    v3 = v5;
    goto LABEL_4;
  }
  for ( i = v5; ; i = *(_DWORD *)(v12 + 552) )
  {
    v12 = v6 + 776LL * i;
    if ( (unsigned __int8)(*(_BYTE *)(v12 + 4) - 6) <= 1u
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v12 + 208) + 168LL) + 109LL) & 0x20) != 0 )
    {
      break;
    }
  }
  v3 = i - 1;
  if ( !v4 )
  {
LABEL_48:
    while ( 1 )
    {
LABEL_4:
      v7 = v6 + 776LL * v3;
      v8 = *(_BYTE *)(v7 + 4);
      if ( v8 != 1 )
      {
        if ( ((v8 - 15) & 0xFD) == 0 || v8 == 2 )
        {
          v9 = 0;
          while ( !a1 || *(_QWORD *)(v7 + v9 + 184) != *(_QWORD *)(a1 + 40) )
          {
            v8 = *(_BYTE *)(v7 + v9 - 772);
            --v3;
            v9 -= 776;
            if ( ((v8 - 15) & 0xFD) != 0 && v8 != 2 )
              goto LABEL_12;
          }
          return (unsigned int)v3;
        }
LABEL_12:
        v3 -= v8 == 9;
        v10 = v6 + 776LL * v3;
        if ( (unsigned __int8)(*(_BYTE *)(v10 + 4) - 6) > 1u
          || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 + 208) + 168LL) + 109LL) & 0x20) == 0 )
        {
          return (unsigned int)-1;
        }
        if ( v4 && v3 < v5 )
        {
          v21 = 776LL * (v3 + 1);
          v22 = v6 + v21;
          if ( *(_BYTE *)(v6 + v21 + 4) == 9 )
            v22 = v6 + v21 + 776;
          v23 = *(_QWORD *)(v22 + 216) >> 3;
          v24 = *qword_4F04C10;
          v25 = *((_DWORD *)qword_4F04C10 + 2);
          while ( 1 )
          {
            v26 = v25 & v23;
            v27 = (_QWORD *)(v24 + 16LL * v26);
            if ( *(_QWORD *)(v22 + 216) == *v27 )
              break;
            if ( !*v27 )
              goto LABEL_42;
            LODWORD(v23) = v26 + 1;
          }
          v28 = v27[1];
          if ( v28 )
            goto LABEL_40;
LABEL_42:
          v29 = *(_QWORD *)(v6 + 776LL * *(int *)(*(_QWORD *)(v22 + 184) + 240LL) + 216);
          v30 = v25 & (v29 >> 3);
          v31 = (__int64 *)(v24 + 16LL * v30);
          v28 = *v31;
          if ( *v31 == v29 )
          {
LABEL_45:
            v28 = v31[1];
          }
          else
          {
            while ( v28 )
            {
              v30 = v25 & (v30 + 1);
              v31 = (__int64 *)(v24 + 16LL * v30);
              v28 = *v31;
              if ( v29 == *v31 )
                goto LABEL_45;
            }
          }
LABEL_40:
          *v4 = v28;
          return (unsigned int)--v3;
        }
      }
      --v3;
    }
  }
  v13 = 776LL * (i + 1);
  v14 = v6 + v13;
  if ( *(_BYTE *)(v6 + v13 + 4) == 9 )
    v14 = v6 + v13 + 776;
  v15 = *(_QWORD *)(v14 + 216);
  for ( j = v15 >> 3; ; LODWORD(j) = v17 + 1 )
  {
    v17 = qword_4F04C10[1] & j;
    v18 = (__int64 *)(*qword_4F04C10 + 16LL * v17);
    v19 = *v18;
    if ( v15 == *v18 )
      break;
    if ( !v19 )
      goto LABEL_30;
  }
  v19 = v18[1];
LABEL_30:
  *v4 = v19;
  return (unsigned int)v3;
}
