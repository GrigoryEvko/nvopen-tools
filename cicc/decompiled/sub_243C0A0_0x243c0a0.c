// Function: sub_243C0A0
// Address: 0x243c0a0
//
bool __fastcall sub_243C0A0(__int64 *a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // rdi
  int v18; // r11d
  unsigned int i; // eax
  __int64 v20; // rcx
  unsigned int v21; // eax
  __int64 v22; // r12
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+10h] [rbp-A0h] BYREF
  char v26; // [rsp+90h] [rbp-20h] BYREF

  v2 = sub_C52410();
  v3 = v2 + 1;
  v4 = sub_C959E0();
  v5 = (_QWORD *)v2[2];
  if ( v5 )
  {
    v6 = v2 + 1;
    do
    {
      while ( 1 )
      {
        v7 = v5[2];
        v8 = v5[3];
        if ( v4 <= v5[4] )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v8 )
          goto LABEL_6;
      }
      v6 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v7 );
LABEL_6:
    if ( v3 != v6 && v4 >= v6[4] )
      v3 = v6;
  }
  if ( v3 == (_QWORD *)((char *)sub_C52410() + 8) )
    return 0;
  v9 = v3[7];
  if ( !v9 )
    return 0;
  v10 = v3 + 6;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v9 + 16);
      v12 = *(_QWORD *)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) >= dword_4FE4608 )
        break;
      v9 = *(_QWORD *)(v9 + 24);
      if ( !v12 )
        goto LABEL_15;
    }
    v10 = (_QWORD *)v9;
    v9 = *(_QWORD *)(v9 + 16);
  }
  while ( v11 );
LABEL_15:
  if ( v3 + 6 == v10 )
    return 0;
  if ( dword_4FE4608 < *((_DWORD *)v10 + 8) )
    return 0;
  if ( !*((_DWORD *)v10 + 9) )
    return 0;
  v14 = *(_QWORD *)(sub_BC1CD0(*a1, &unk_4F82410, a1[1]) + 8);
  v15 = *(_QWORD *)(a1[1] + 40);
  v16 = *(unsigned int *)(v14 + 88);
  v17 = *(_QWORD *)(v14 + 72);
  if ( !(_DWORD)v16 )
    return 0;
  v18 = 1;
  for ( i = (v16 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)))); ; i = (v16 - 1) & v21 )
  {
    v20 = v17 + 24LL * i;
    if ( *(_UNKNOWN **)v20 == &unk_4F87C68 && v15 == *(_QWORD *)(v20 + 8) )
      break;
    if ( *(_QWORD *)v20 == -4096 && *(_QWORD *)(v20 + 8) == -4096 )
      return 0;
    v21 = v18 + i;
    ++v18;
  }
  if ( v20 == v17 + 24 * v16 )
    return 0;
  v22 = *(_QWORD *)(*(_QWORD *)(v20 + 16) + 24LL);
  if ( !v22 )
    return 0;
  v23 = &v25;
  do
  {
    *v23 = -4096;
    v23 += 2;
  }
  while ( v23 != (__int64 *)&v26 );
  if ( !*(_QWORD *)(v22 + 16) )
    return 0;
  v24 = sub_BC1CD0(*a1, &unk_4F8D9A8, a1[1]);
  return sub_243BED0(v22 + 8, qword_4FE4688, a1[1], (__int64 *)(v24 + 8));
}
