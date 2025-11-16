// Function: sub_8C7A50
// Address: 0x8c7a50
//
_BOOL8 __fastcall sub_8C7A50(__int64 a1)
{
  _BOOL4 v1; // r14d
  __int64 *v2; // rax
  __int64 v3; // r13
  unsigned __int8 v5; // al
  __int64 *v6; // rdx
  __int64 v7; // rax
  __m128i *v8; // rbx
  __m128i *v9; // rax
  _UNKNOWN *__ptr32 *v10; // r8
  unsigned int v11; // edx

  v1 = 1;
  v2 = *(__int64 **)(a1 + 32);
  if ( v2 )
  {
    v3 = *v2;
    if ( a1 == *v2 )
    {
      v3 = v2[1];
      if ( !v3 || a1 == v3 )
        return 1;
    }
    v1 = sub_8C7610(a1);
    if ( !v1 )
      goto LABEL_4;
    if ( !(unsigned int)sub_8DED30(*(_QWORD *)(a1 + 120), *(_QWORD *)(v3 + 120), 261)
      || !(unsigned int)sub_8DBAE0(*(_QWORD *)(a1 + 120), *(_QWORD *)(v3 + 120)) )
    {
      goto LABEL_28;
    }
    v5 = *(_BYTE *)(a1 + 172);
    if ( ((v5 ^ *(_BYTE *)(v3 + 172)) & 4) != 0 )
    {
      if ( (*(_BYTE *)(a1 + 170) & 0x10) == 0 )
        goto LABEL_28;
    }
    else if ( (v5 & 4) != 0 )
    {
      v8 = sub_740200(a1);
      v9 = sub_740200(v3);
      v11 = 0;
      if ( !dword_4D04964 )
      {
        v11 = 16;
        if ( (*(_BYTE *)(a1 + 170) & 0x20) != 0 )
          v11 = (*(_BYTE *)(v3 + 170) & 0x20) == 0 ? 16 : 48;
      }
      if ( !(unsigned int)sub_739430((__int64)v8, (__int64)v9, v11, (unsigned int)dword_4D04964, v10) )
        goto LABEL_28;
    }
    if ( ((*(_BYTE *)(v3 + 88) ^ *(_BYTE *)(a1 + 88)) & 0x73) == 0 )
    {
      if ( !unk_4D04958
        && !*(_BYTE *)(a1 + 136)
        && !*(_BYTE *)(v3 + 136)
        && (*(_BYTE *)(a1 + 172) & 0x20) == 0
        && (*(_BYTE *)(a1 + 170) & 0x90) != 0x10
        && (*(_BYTE *)(a1 + 168) & 8) == 0
        && (*(_BYTE *)(v3 + 168) & 8) == 0
        && (dword_4F077C4 == 2 || *(_BYTE *)(a1 + 177) && *(_BYTE *)(v3 + 177)) )
      {
        v6 = *(__int64 **)(a1 + 32);
        v7 = a1;
        if ( v6 )
          v7 = *v6;
        sub_8C6700((__int64 *)a1, (unsigned int *)(v7 + 64), 0x433u, 0x434u);
      }
      goto LABEL_4;
    }
LABEL_28:
    sub_8C6700((__int64 *)a1, (unsigned int *)(v3 + 64), 0x42Au, 0x425u);
    v1 = 0;
    sub_8C7090(7, a1);
LABEL_4:
    sub_8C6CA0(a1, v3, 7u, (_QWORD *)(v3 + 64));
    sub_8C6CA0(v3, a1, 7u, (_QWORD *)(a1 + 64));
  }
  return v1;
}
