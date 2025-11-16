// Function: sub_6E8160
// Address: 0x6e8160
//
__int64 __fastcall sub_6E8160(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        _DWORD *a9)
{
  int v9; // r13d
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r14
  char v17; // dl
  __int64 v18; // rax
  char v19; // al
  bool v20; // zf
  int v22; // eax
  __int64 v23; // rax
  unsigned int v24; // [rsp+4h] [rbp-4Ch]
  char v25; // [rsp+8h] [rbp-48h]
  int v26; // [rsp+Ch] [rbp-44h]
  int v27; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  v9 = a5;
  v26 = a3;
  v24 = a4;
  v25 = a6;
  v11 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v12 = *a1;
  v28[0] = v11;
  v13 = *(_QWORD *)v12;
  if ( a8 )
  {
    v20 = (unsigned int)sub_6E6010() == 0;
    v22 = 0;
    if ( !v20 )
      v22 = v26;
    v26 = v22;
  }
  v14 = a2;
  v15 = sub_73D720(a2);
  v16 = v15;
  if ( !v9 || (a2 = v15, v14 = v13, !(unsigned int)sub_8DAAE0(v13, v15)) || (v14 = v12, (unsigned int)sub_6DEAC0(v12)) )
  {
    v17 = *(_BYTE *)(v16 + 140);
    if ( v17 == 12 )
    {
      v18 = v16;
      do
      {
        v18 = *(_QWORD *)(v18 + 160);
        v17 = *(_BYTE *)(v18 + 140);
      }
      while ( v17 == 12 );
    }
    if ( !v17 )
      goto LABEL_15;
    while ( 1 )
    {
      v19 = *(_BYTE *)(v13 + 140);
      if ( v19 != 12 )
        break;
      v13 = *(_QWORD *)(v13 + 160);
    }
    if ( v19 || (v14 = v16, !(unsigned int)sub_8D3A70(v16)) )
    {
      v20 = *(_BYTE *)(v12 + 24) == 2;
      v27 = 1;
      if ( v20 )
      {
        sub_72A510(*(_QWORD *)(v12 + 56), v28[0]);
        if ( a7 && dword_4D0488C && (!dword_4F077BC || (_DWORD)qword_4F077B4 || !qword_4F077A8) )
        {
          v27 = 1;
        }
        else
        {
          if ( a8 )
            sub_6E5170(v28[0], v16, v9, v26, v24, a7, 0, (__int64)&v27, (__int64)a9);
          else
            sub_7115B0(v28[0], v16, v9, 0, 1, 0, 0, v26, v24, a7, 0, (__int64)&v27, 0, (__int64)a9);
          if ( !v27 )
          {
            v23 = sub_73A460(v28[0]);
            *(_QWORD *)(v12 + 56) = v23;
            *(_BYTE *)(v23 + 168) = ((v25 & 1) << 6) | *(_BYTE *)(v23 + 168) & 0xBF;
            *(_QWORD *)v12 = v16;
            return sub_724E30(v28);
          }
        }
      }
      sub_6E7AE0(a1, v16, v26, v24, v9, v25, a7, a9);
    }
    else
    {
LABEL_15:
      *a1 = sub_7305B0(v14, a2);
    }
  }
  return sub_724E30(v28);
}
