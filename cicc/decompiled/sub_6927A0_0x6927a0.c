// Function: sub_6927A0
// Address: 0x6927a0
//
__int64 __fastcall sub_6927A0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4, int a5, __int64 a6)
{
  unsigned __int8 v11; // al
  __int64 v12; // r10
  __int64 v13; // rax
  char k; // dl
  int v15; // edx
  __int64 v16; // rax
  __int64 v18; // rdi
  int v19; // r8d
  __int64 i; // rax
  __int64 j; // rax
  int v22; // esi
  __int64 v23; // rax
  __int64 v24; // [rsp+0h] [rbp-60h]
  __int64 v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  unsigned __int8 v29; // [rsp+18h] [rbp-48h]
  int v30[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v30[0] = 0;
  if ( dword_4F077C4 != 2 || !a5 )
    goto LABEL_41;
  if ( (unsigned int)sub_68FE10(a1, 0, 1) || (unsigned int)sub_68FE10((_BYTE *)a2, 0, 1) )
  {
    if ( !(unsigned int)sub_8D3A70(*a1) )
      goto LABEL_29;
    v18 = *a1;
    if ( dword_4F077C4 == 2 )
    {
      if ( (unsigned int)sub_8D23B0(v18) )
        sub_8AE000(v18);
      v18 = *a1;
    }
    v19 = sub_8D23B0(v18);
    if ( qword_4D0495C )
    {
      for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 177LL) & 0x20) != 0 )
        v19 = 1;
    }
    if ( *(_BYTE *)(a2 + 16) == 5 )
    {
      for ( j = *a1; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)j + 96LL) + 177LL) & 0x20) != 0 )
LABEL_29:
        v19 = 1;
    }
    sub_84EC30(15, 0, 1, 0, v19, (_DWORD)a1, a2, a3, a4, 0, 0, a6, 0, 0, (__int64)v30);
  }
  if ( !v30[0] )
  {
LABEL_41:
    if ( !(unsigned int)sub_6E9250(a3) )
    {
      if ( v30[0] )
        goto LABEL_15;
      sub_6F69D0(a1, 4);
      if ( (unsigned int)sub_702F90(a1) )
      {
        sub_6ECF90(a1, 0);
        goto LABEL_7;
      }
      if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
      {
LABEL_7:
        v25 = *a1;
        v26 = sub_73D720(*a1);
        v11 = sub_6E9930(56, v26);
        v12 = v26;
        v29 = v11;
        if ( *(_BYTE *)(a2 + 16) == 5 )
        {
          v22 = v26;
          v24 = v26;
          v28 = *(_QWORD *)(a2 + 144);
          sub_839D30(v28, v22, 0, 1, 0, 0, 1, 0, 0, a2, 0, 0);
          sub_6E1990(v28);
          v12 = v24;
        }
        v27 = v12;
        sub_847710(a2, v12, 513, a3);
        sub_6F7CB0(a1, a2, v29, v27, a6);
        if ( dword_4F077C4 == 2 )
        {
          if ( !*(_BYTE *)(a6 + 16) )
            goto LABEL_14;
          v13 = *(_QWORD *)a6;
          for ( k = *(_BYTE *)(*(_QWORD *)a6 + 140LL); k == 12; k = *(_BYTE *)(v13 + 140) )
            v13 = *(_QWORD *)(v13 + 160);
          if ( k )
          {
            v23 = *(_QWORD *)(a6 + 144);
            *(_BYTE *)(v23 + 25) |= 1u;
            *(_BYTE *)(v23 + 58) |= 1u;
            *(_QWORD *)a6 = v25;
            *(_QWORD *)v23 = v25;
            *(_QWORD *)(a6 + 88) = a1[11];
            sub_6E6A20(a6);
          }
          else
          {
LABEL_14:
            sub_6E6870(a6);
          }
        }
        goto LABEL_15;
      }
    }
    sub_6E6260(a6);
  }
LABEL_15:
  v15 = *((_DWORD *)a1 + 17);
  *(_WORD *)(a6 + 72) = *((_WORD *)a1 + 36);
  *(_DWORD *)(a6 + 68) = v15;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a6 + 68);
  v16 = *(_QWORD *)(a2 + 76);
  *(_QWORD *)(a6 + 76) = v16;
  unk_4F061D8 = v16;
  sub_6E3280(a6, a3);
  return sub_6E3BA0(a6, a3, a4, 0);
}
