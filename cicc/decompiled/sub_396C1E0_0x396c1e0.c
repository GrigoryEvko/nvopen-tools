// Function: sub_396C1E0
// Address: 0x396c1e0
//
__int64 __fastcall sub_396C1E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 *v3; // rdx
  unsigned int v5; // r12d
  _QWORD *v7; // r15
  unsigned int v8; // eax
  _QWORD *v9; // r13
  unsigned __int64 i; // r14
  __int16 v11; // ax
  char v12; // al
  __int16 v13; // ax
  __int64 v14; // rax
  __int16 v15; // r8
  unsigned __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rcx

  if ( *(_BYTE *)(a2 + 180) )
    return 0;
  v2 = *(__int64 **)(a2 + 72);
  v3 = *(__int64 **)(a2 + 64);
  if ( v3 == v2 )
    return 0;
  if ( (unsigned int)(v2 - v3) > 1 )
    return 0;
  v7 = (_QWORD *)*v3;
  LOBYTE(v8) = sub_1DD69A0(*v3, a2);
  v5 = v8;
  if ( !(_BYTE)v8 )
    return 0;
  v9 = v7 + 3;
  if ( v7 + 3 == (_QWORD *)(v7[3] & 0xFFFFFFFFFFFFFFF8LL) )
    return v5;
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 232) + 504LL) - 34) <= 1
    && (unsigned int)((__int64)(v7[12] - v7[11]) >> 3) > 2 )
  {
    return 0;
  }
  else
  {
    for ( i = sub_1DD5EE0((__int64)v7); v9 != (_QWORD *)i; i = *(_QWORD *)(i + 8) )
    {
      v11 = *(_WORD *)(i + 46);
      if ( (v11 & 4) == 0 && (v11 & 8) != 0 )
        v12 = sub_1E15D00(i, 0x80u, 1);
      else
        v12 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(i + 16) + 8LL) >> 7;
      if ( !v12 )
        return 0;
      v13 = *(_WORD *)(i + 46);
      if ( (v13 & 4) == 0 && (v13 & 8) != 0 )
        LOBYTE(v14) = sub_1E15D00(i, 0x100u, 1);
      else
        v14 = (*(_QWORD *)(*(_QWORD *)(i + 16) + 8LL) >> 8) & 1LL;
      if ( (_BYTE)v14 )
        return 0;
      v15 = *(_WORD *)(i + 46);
      v16 = i;
      if ( (v15 & 4) != 0 )
      {
        do
          v16 = *(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v16 + 46) & 4) != 0 );
      }
      v17 = *(_QWORD *)(i + 24) + 24LL;
      while ( 1 )
      {
        v18 = *(_QWORD *)(v16 + 32);
        v19 = v18 + 40LL * *(unsigned int *)(v16 + 40);
        if ( v18 != v19 )
          break;
        v16 = *(_QWORD *)(v16 + 8);
        if ( v17 == v16 || (*(_BYTE *)(v16 + 46) & 4) == 0 )
          goto LABEL_34;
      }
      do
      {
        while ( 1 )
        {
          if ( *(_BYTE *)v18 == 8 || *(_BYTE *)v18 == 4 && a2 == *(_QWORD *)(v18 + 24) )
            return 0;
          v20 = v18 + 40;
          v21 = v19;
          if ( v20 == v19 )
            break;
          v19 = v20;
LABEL_40:
          v18 = v19;
          v19 = v21;
        }
        while ( 1 )
        {
          v16 = *(_QWORD *)(v16 + 8);
          if ( v17 == v16 || (*(_BYTE *)(v16 + 46) & 4) == 0 )
            break;
          v19 = *(_QWORD *)(v16 + 32);
          v21 = v19 + 40LL * *(unsigned int *)(v16 + 40);
          if ( v19 != v21 )
            goto LABEL_40;
        }
        v18 = v19;
        v19 = v21;
LABEL_34:
        ;
      }
      while ( v18 != v19 );
      if ( (*(_BYTE *)i & 4) == 0 && (v15 & 8) != 0 )
      {
        do
          i = *(_QWORD *)(i + 8);
        while ( (*(_BYTE *)(i + 46) & 8) != 0 );
      }
    }
  }
  return v5;
}
