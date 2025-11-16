// Function: sub_1BAFE70
// Address: 0x1bafe70
//
char __fastcall sub_1BAFE70(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // rax
  __int64 v7; // rax
  __int64 v8; // rcx
  int v9; // edx
  _BYTE *v10; // rsi
  int v11; // edi
  int v12; // eax
  int v13; // eax
  _BYTE *v14; // rdx
  __int64 v15; // r13
  __int64 i; // rbx
  unsigned __int8 v17; // al
  _BYTE *v19; // [rsp-60h] [rbp-60h] BYREF
  __int64 v20; // [rsp-58h] [rbp-58h] BYREF

  LOBYTE(v3) = a3[16];
  if ( (unsigned __int8)v3 <= 0x17u )
    return v3;
  if ( (_BYTE)v3 == 71 )
  {
    v3 = *(_QWORD *)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) != 15 )
      return v3;
  }
  else if ( (_BYTE)v3 != 56 )
  {
    return v3;
  }
  LOBYTE(v3) = sub_13FC1A0(*(_QWORD *)(**(_QWORD **)a1 + 296LL), (__int64)a3);
  if ( (_BYTE)v3 )
    return v3;
  v7 = *(_QWORD *)(a1 + 8);
  v19 = a3;
  if ( (*(_BYTE *)(v7 + 8) & 1) != 0 )
  {
    v8 = v7 + 16;
    v9 = 7;
  }
  else
  {
    v8 = *(_QWORD *)(v7 + 16);
    v12 = *(_DWORD *)(v7 + 24);
    if ( !v12 )
      goto LABEL_17;
    v9 = v12 - 1;
  }
  LODWORD(v3) = v9 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v10 = *(_BYTE **)(v8 + 8LL * (unsigned int)v3);
  if ( a3 == v10 )
    return v3;
  v11 = 1;
  while ( v10 != (_BYTE *)-8LL )
  {
    LODWORD(v3) = v9 & (v11 + v3);
    v10 = *(_BYTE **)(v8 + 8LL * (unsigned int)v3);
    if ( a3 == v10 )
      return v3;
    ++v11;
  }
LABEL_17:
  v13 = sub_1B99570(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL), a2, ***(_DWORD ***)(a1 + 16));
  if ( *(_BYTE *)(a2 + 16) == 55 )
  {
    v14 = *(_BYTE **)(a2 - 48);
    if ( v14 )
    {
      if ( a3 == v14 )
      {
        if ( v13 == 5 )
          goto LABEL_21;
LABEL_27:
        v15 = (__int64)v19;
LABEL_28:
        LOBYTE(v3) = sub_165A590((__int64)&v20, *(_QWORD *)(a1 + 32), v15);
        return v3;
      }
    }
  }
  if ( v13 == 4 )
    goto LABEL_27;
LABEL_21:
  v15 = (__int64)v19;
  for ( i = *((_QWORD *)v19 + 1); i; i = *(_QWORD *)(i + 8) )
  {
    v17 = *((_BYTE *)sub_1648700(i) + 16);
    if ( v17 <= 0x17u || (unsigned __int8)(v17 - 54) > 1u )
      goto LABEL_28;
  }
  LOBYTE(v3) = sub_1BAFD60(*(_QWORD *)(a1 + 24), (__int64 *)&v19);
  return v3;
}
