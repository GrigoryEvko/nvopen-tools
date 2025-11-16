// Function: sub_3239A30
// Address: 0x3239a30
//
__int64 __fastcall sub_3239A30(__int64 a1, __int64 *a2, unsigned int a3)
{
  __int64 *v3; // r14
  __int64 *v4; // rax
  __int64 v5; // r12
  __int64 (*v6)(void); // rax
  bool v7; // bl
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 i; // r15
  __int64 v11; // rax
  __int16 v12; // si
  char v13; // dl
  __int64 (__fastcall *v14)(__int64); // rax
  char v15; // si
  char v16; // al
  int v17; // eax
  __int64 v18; // r12
  unsigned __int8 v19; // al
  __int64 v20; // rdx
  __int64 v21; // rbx
  unsigned __int16 v22; // ax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v31; // [rsp+8h] [rbp-68h]
  char v33; // [rsp+17h] [rbp-59h]
  char v34; // [rsp+17h] [rbp-59h]
  _BYTE v36[80]; // [rsp+20h] [rbp-50h] BYREF

  v3 = a2 + 40;
  v4 = (__int64 *)a2[41];
  if ( v4 == a2 + 40 )
    return 0;
  while ( (__int64 *)(v4[6] & 0xFFFFFFFFFFFFFFF8LL) == v4 + 6 )
  {
    v4 = (__int64 *)v4[1];
    if ( v4 == v3 )
      return 0;
  }
  if ( v3 == v4 )
    return 0;
  v5 = 0;
  v6 = *(__int64 (**)(void))(*(_QWORD *)a2[2] + 128LL);
  if ( v6 != sub_2DAC790 )
    v5 = v6();
  v7 = 0;
  v8 = *a2;
  if ( (*(_BYTE *)(*a2 + 2) & 4) == 0 )
  {
    v7 = 1;
    if ( (*(_BYTE *)(v8 + 7) & 0x20) != 0 )
      v7 = sub_B91C10(v8, 32) == 0;
  }
  v9 = a2[41];
  for ( i = *(_QWORD *)(v9 + 56); v9 + 48 == (*(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL); i = *(_QWORD *)(v9 + 56) )
    v9 = *(_QWORD *)(v9 + 8);
  v31 = 0;
  while ( 1 )
  {
    v11 = *(_QWORD *)(i + 16);
    if ( (*(_BYTE *)(v11 + 24) & 0x10) != 0 )
      goto LABEL_35;
    v12 = *(_WORD *)(i + 68);
    if ( v12 == 20 )
    {
      v13 = 1;
    }
    else
    {
      v13 = 0;
      v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 520LL);
      if ( v14 != sub_2DCA430 )
      {
        ((void (__fastcall *)(_BYTE *, __int64, __int64))v14)(v36, v5, i);
        v13 = v36[16];
        v12 = *(_WORD *)(i + 68);
      }
      if ( v12 == 10 )
      {
        v15 = 1;
        if ( (*(_DWORD *)(i + 40) & 0xFFFFFF) == 1 )
          goto LABEL_21;
      }
      v11 = *(_QWORD *)(i + 16);
    }
    v15 = 0;
    if ( (*(_BYTE *)(v11 + 27) & 0x20) != 0 )
    {
      v33 = v13;
      v16 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v5 + 56LL))(v5, i);
      v13 = v33;
      v15 = v16;
    }
LABEL_21:
    if ( (*(_BYTE *)(i + 44) & 1) == 0 )
    {
      if ( *(_QWORD *)(i + 56) )
      {
        v34 = v13;
        v17 = sub_B10CE0(i + 56);
        v13 = v34;
        if ( v17 )
          goto LABEL_24;
      }
      if ( !v15 && !v13 )
      {
        v23 = v31;
        if ( !v31 )
          v23 = i;
        v31 = v23;
      }
    }
    v7 = 0;
LABEL_35:
    v24 = i;
    if ( (*(_BYTE *)i & 4) == 0 && (*(_BYTE *)(i + 44) & 8) != 0 )
    {
      do
        v24 = *(_QWORD *)(v24 + 8);
      while ( (*(_BYTE *)(v24 + 44) & 8) != 0 );
    }
    v25 = *(_QWORD *)(v24 + 8);
    if ( v25 == *(_QWORD *)(i + 24) + 48LL )
      break;
LABEL_39:
    i = v25;
  }
  v26 = *(_DWORD *)(i + 44);
  if ( (v26 & 4) != 0 || (v26 & 8) == 0 )
    v27 = (*(_QWORD *)(*(_QWORD *)(i + 16) + 24LL) >> 9) & 1LL;
  else
    LOBYTE(v27) = sub_2E88A90(i, 512, 1);
  if ( !(_BYTE)v27 && *(_DWORD *)(v9 + 72) <= 1u )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v9 + 8);
      if ( v3 == (__int64 *)v9 )
        break;
      if ( v9 + 48 != (*(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v25 = *(_QWORD *)(v9 + 56);
        goto LABEL_39;
      }
    }
  }
  i = 0;
  if ( v31 )
  {
    v28 = a2[41];
    if ( *(_QWORD *)(v31 + 24) == v28 )
    {
      i = v31;
      v7 = *(_QWORD *)(v28 + 56) == v31;
    }
  }
LABEL_24:
  if ( v7 && (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL) + 544LL) - 42) > 1 )
  {
    if ( !i || *(_QWORD *)(i + 56) && !*(_DWORD *)(sub_B10CD0(i + 56) + 4) )
    {
      i = 0;
      goto LABEL_26;
    }
  }
  else
  {
LABEL_26:
    v18 = sub_B92180(*a2);
    v19 = *(_BYTE *)(v18 - 16);
    if ( (v19 & 2) != 0 )
      v20 = *(_QWORD *)(v18 - 32);
    else
      v20 = v18 - 16 - 8LL * ((v19 >> 2) & 0xF);
    sub_3238860(a1, *(_QWORD *)(v20 + 40));
    v21 = *(_QWORD *)(a1 + 3232);
    v22 = sub_3220AA0(a1);
    sub_321C5E0(*(_QWORD ***)(a1 + 8), *(_DWORD *)(v18 + 20), 0, (_BYTE *)v18, 1u, a3, v22, v21, 0, 0);
  }
  return i;
}
