// Function: sub_3983AE0
// Address: 0x3983ae0
//
void __fastcall sub_3983AE0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rbx
  int v5; // eax
  __int64 v6; // rdi
  __int64 v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rax
  char v10; // si
  __int64 v11; // rax
  void (__fastcall *v12)(__int64, __int64, __int64 (*)(void)); // rax
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rsi
  int v16; // eax
  int v17; // [rsp+Ch] [rbp-44h]
  __int64 v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  *(_BYTE *)(a1 + 26) = 0;
  *(_WORD *)(a1 + 28) = 0;
  v4 = *a2;
  v18 = a2[52];
  v19 = a2[51];
  v5 = sub_396EB00(*(_QWORD *)(a1 + 8));
  v6 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 29) = v5 != 0;
  v7 = sub_396DD80(v6);
  v17 = *(_DWORD *)(v7 + 12);
  if ( (*(_BYTE *)(v4 + 18) & 8) == 0 )
  {
    *(_BYTE *)(a1 + 27) = 0;
    goto LABEL_3;
  }
  v13 = sub_15E38F0(v4);
  v14 = sub_1649C60(v13);
  if ( *(_BYTE *)(v14 + 16) )
    v14 = 0;
  if ( (*(_BYTE *)(v4 + 18) & 8) != 0
    && !(unsigned int)sub_14DD7D0(v14)
    && ((unsigned __int8)sub_1560180(v4 + 112, 56)
     || !(unsigned __int8)sub_1560180(v4 + 112, 30)
     || (*(_BYTE *)(v4 + 18) & 8) != 0) )
  {
    *(_BYTE *)(a1 + 27) = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 27) = 0;
    if ( v17 == 255 || v19 == v18 )
      goto LABEL_3;
  }
  if ( !v14 )
  {
LABEL_3:
    *(_BYTE *)(a1 + 26) = 0;
    *(_BYTE *)(a1 + 28) = 0;
    v8 = *(_QWORD *)(a2[4] + 184);
    if ( (*(_DWORD *)(v8 + 348) & 0xFFFFFFFD) == 1 )
      goto LABEL_4;
    v10 = 0;
    if ( *(_DWORD *)(v8 + 348) != 4 )
      goto LABEL_7;
    goto LABEL_28;
  }
  *(_BYTE *)(a1 + 26) = 1;
  *(_BYTE *)(a1 + 28) = *(_DWORD *)(v7 + 16) != 255;
  v8 = *(_QWORD *)(a2[4] + 184);
  if ( (*(_DWORD *)(v8 + 348) & 0xFFFFFFFD) == 1 )
    goto LABEL_19;
  v10 = 1;
  if ( *(_DWORD *)(v8 + 348) != 4 )
    goto LABEL_7;
LABEL_28:
  v16 = *(_DWORD *)(v8 + 352);
  if ( !v16 || v16 == 6 )
    goto LABEL_7;
  if ( v10 )
  {
LABEL_19:
    *(_BYTE *)(a1 + 24) = 1;
    v9 = *(_QWORD *)a1;
LABEL_20:
    v12 = *(void (__fastcall **)(__int64, __int64, __int64 (*)(void)))(v9 + 56);
    v15 = a2[41];
    if ( v12 == sub_3983950 )
    {
      sub_3983800(a1, v15, (__int64 (*)(void))sub_39837B0);
      return;
    }
    goto LABEL_33;
  }
LABEL_4:
  if ( *(_BYTE *)(a1 + 29) )
  {
    *(_BYTE *)(a1 + 24) = 1;
    v9 = *(_QWORD *)a1;
    goto LABEL_20;
  }
LABEL_7:
  v11 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 24) = 0;
  v12 = *(void (__fastcall **)(__int64, __int64, __int64 (*)(void)))(v11 + 56);
  if ( v12 != sub_3983950 )
  {
    v15 = a2[41];
LABEL_33:
    v12(a1, v15, (__int64 (*)(void))sub_39837B0);
  }
}
