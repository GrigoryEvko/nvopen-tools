// Function: sub_38528E0
// Address: 0x38528e0
//
__int64 __fastcall sub_38528E0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned __int8 *v6; // rsi
  unsigned __int8 *v7; // rdx
  char v8; // al
  __int64 v9; // rcx
  int v10; // edi
  int v11; // ecx
  _QWORD *v12; // rdx
  unsigned int v13; // r13d
  int v15; // eax
  int v16; // eax
  __int64 v17; // rcx
  unsigned int v18; // esi
  __int64 *v19; // rdx
  __int64 v20; // rdi
  unsigned int v21; // edi
  __int64 *v22; // rdx
  __int64 v23; // r8
  int v24; // eax
  int v25; // edx
  int v26; // edx
  int v27; // r9d
  int v28; // r8d
  __int64 v29[12]; // [rsp+10h] [rbp-60h] BYREF

  v5 = *((_QWORD *)a2 - 3);
  v6 = (unsigned __int8 *)*((_QWORD *)a2 - 6);
  v4 = (__int64)v6;
  if ( v6[16] <= 0x10u )
    goto LABEL_2;
  v15 = *(_DWORD *)(a1 + 160);
  if ( !v15 )
  {
    v6 = 0;
LABEL_2:
    v7 = (unsigned __int8 *)v5;
    if ( *(_BYTE *)(v5 + 16) <= 0x10u )
      goto LABEL_3;
    v24 = *(_DWORD *)(a1 + 160);
    v17 = *(_QWORD *)(a1 + 144);
    v7 = 0;
    if ( !v24 )
      goto LABEL_3;
    v16 = v24 - 1;
    goto LABEL_30;
  }
  v16 = v15 - 1;
  v17 = *(_QWORD *)(a1 + 144);
  v18 = v16 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v19 = (__int64 *)(v17 + 16LL * v18);
  v20 = *v19;
  if ( v4 == *v19 )
  {
LABEL_28:
    v6 = (unsigned __int8 *)v19[1];
  }
  else
  {
    v26 = 1;
    while ( v20 != -8 )
    {
      v28 = v26 + 1;
      v18 = v16 & (v26 + v18);
      v19 = (__int64 *)(v17 + 16LL * v18);
      v20 = *v19;
      if ( v4 == *v19 )
        goto LABEL_28;
      v26 = v28;
    }
    v6 = 0;
  }
  v7 = (unsigned __int8 *)v5;
  if ( *(_BYTE *)(v5 + 16) > 0x10u )
  {
LABEL_30:
    v21 = v16 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v22 = (__int64 *)(v17 + 16LL * v21);
    v23 = *v22;
    if ( *v22 == v5 )
    {
LABEL_31:
      v7 = (unsigned __int8 *)v22[1];
    }
    else
    {
      v25 = 1;
      while ( v23 != -8 )
      {
        v27 = v25 + 1;
        v21 = v16 & (v25 + v21);
        v22 = (__int64 *)(v17 + 16LL * v21);
        v23 = *v22;
        if ( v5 == *v22 )
          goto LABEL_31;
        v25 = v27;
      }
      v7 = 0;
    }
  }
LABEL_3:
  v8 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v8 == 16 )
    v8 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
  v9 = *(_QWORD *)(a1 + 40);
  memset(&v29[1], 0, 32);
  v10 = a2[16];
  v29[0] = v9;
  if ( (unsigned __int8)(v8 - 1) <= 5u || (_BYTE)v10 == 76 )
  {
    v11 = a2[17] >> 1;
    if ( v11 == 127 )
      LOBYTE(v11) = -1;
    if ( !v7 )
      v7 = (unsigned __int8 *)v5;
    if ( !v6 )
      v6 = (unsigned __int8 *)v4;
    v12 = sub_13E1150(v10 - 24, v6, v7, v11, v29);
    if ( !v12 )
      goto LABEL_22;
  }
  else
  {
    if ( !v7 )
      v7 = (unsigned __int8 *)v5;
    if ( !v6 )
      v6 = (unsigned __int8 *)v4;
    v12 = sub_13E1140(v10 - 24, v6, v7, v29);
    if ( !v12 )
    {
LABEL_22:
      sub_384F350(a1, v4);
      sub_384F350(a1, v5);
      if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)a2 + 8LL) - 1) <= 5u && (unsigned int)sub_14A3050(*(_QWORD *)a1) == 4 )
        *(_DWORD *)(a1 + 76) += 25;
      return 0;
    }
  }
  v13 = 1;
  if ( *((_BYTE *)v12 + 16) <= 0x10u )
  {
    v29[0] = (__int64)a2;
    sub_38526A0(a1 + 136, v29)[1] = v12;
  }
  return v13;
}
