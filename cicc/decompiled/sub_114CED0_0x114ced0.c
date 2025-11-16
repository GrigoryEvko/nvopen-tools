// Function: sub_114CED0
// Address: 0x114ced0
//
_BYTE *__fastcall sub_114CED0(__int64 a1, __int64 a2, unsigned __int8 a3, int a4)
{
  char v6; // dl
  __int64 v7; // rax
  _BYTE *result; // rax
  bool v9; // r14
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rdx
  _BYTE *v17; // rsi
  __int64 v18; // rax
  bool v19; // al
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-48h]
  int v28; // [rsp+8h] [rbp-48h]
  __int64 v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 != 86 )
    goto LABEL_2;
  v17 = *(_BYTE **)(a2 - 64);
  result = *(_BYTE **)(a2 - 32);
  if ( *v17 != 20 )
  {
    if ( *result == 20 )
      return v17;
LABEL_2:
    v7 = *(_QWORD *)(a2 + 16);
    if ( !v7 )
      return 0;
    v9 = a4 == 3 || *(_QWORD *)(v7 + 8) != 0;
    if ( v9 )
      return 0;
    if ( v6 == 63 )
    {
      if ( a3 || (v28 = a4, v19 = sub_B4DE30(a2), a4 = v28, v19) )
      {
        v18 = sub_114CED0(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a3, (unsigned int)(a4 + 1));
        if ( v18 )
        {
          if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
            v20 = *(_QWORD *)(a2 - 8);
          else
            v20 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
          v21 = *(_QWORD *)v20;
          if ( *(_QWORD *)v20 )
          {
            v22 = *(_QWORD *)(v20 + 8);
            **(_QWORD **)(v20 + 16) = v22;
            if ( v22 )
              *(_QWORD *)(v22 + 16) = *(_QWORD *)(v20 + 16);
          }
          *(_QWORD *)v20 = v18;
          v23 = *(_QWORD *)(v18 + 16);
          *(_QWORD *)(v20 + 8) = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = v20 + 8;
          *(_QWORD *)(v20 + 16) = v18 + 16;
          *(_QWORD *)(v18 + 16) = v20;
          if ( *(_BYTE *)v21 > 0x1Cu )
          {
            v24 = *(_QWORD *)(a1 + 40);
            v29[0] = v21;
            v25 = v24 + 2096;
            sub_114BD80(v24 + 2096, v29);
            v26 = *(_QWORD *)(v21 + 16);
            if ( v26 )
            {
              if ( !*(_QWORD *)(v26 + 8) )
              {
                v29[0] = *(_QWORD *)(v26 + 24);
                sub_114BD80(v25, v29);
              }
            }
          }
LABEL_21:
          sub_F15FC0(*(_QWORD *)(a1 + 40), a2);
          return 0;
        }
      }
      v6 = *(_BYTE *)a2;
    }
    if ( v6 != 84 )
      return 0;
    v10 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v11 = *(_QWORD *)(a2 - 8);
      v27 = v11 + v10;
    }
    else
    {
      v27 = a2;
      v11 = a2 - v10;
    }
    if ( v27 == v11 )
      return 0;
    v12 = (__int64 *)v11;
    do
    {
      v13 = sub_114CED0(a1, *v12, a3, 3);
      if ( v13 )
      {
        v14 = *v12;
        if ( *v12 )
        {
          v15 = v12[1];
          *(_QWORD *)v12[2] = v15;
          if ( v15 )
            *(_QWORD *)(v15 + 16) = v12[2];
        }
        *v12 = v13;
        v16 = *(_QWORD *)(v13 + 16);
        v12[1] = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = v12 + 1;
        v12[2] = v13 + 16;
        v9 = 1;
        *(_QWORD *)(v13 + 16) = v12;
        sub_114C030(*(_QWORD *)(a1 + 40), v14);
      }
      v12 += 4;
    }
    while ( (__int64 *)v27 != v12 );
    if ( !v9 )
      return 0;
    goto LABEL_21;
  }
  return result;
}
