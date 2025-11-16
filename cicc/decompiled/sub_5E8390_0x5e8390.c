// Function: sub_5E8390
// Address: 0x5e8390
//
__int64 __fastcall sub_5E8390(__int64 a1, char a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // rax
  int i; // r12d
  int v11; // eax
  int v12; // esi
  int v13; // r15d
  __int64 v15; // rax
  int v16; // eax
  int v17; // [rsp+Ch] [rbp-54h]
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  _BYTE v20[52]; // [rsp+2Ch] [rbp-34h] BYREF

  v6 = a1;
  for ( i = a1; *(_BYTE *)(v6 + 140) == 12; v6 = *(_QWORD *)(v6 + 160) )
    ;
  v19 = *(_QWORD *)(*(_QWORD *)v6 + 96LL);
  if ( a3 )
  {
    v17 = a4;
    v18 = *(_QWORD *)(a3 + 8);
    v11 = sub_8D32E0(v18);
    v12 = 0;
    a1 = v18;
    a4 = v17;
    v13 = v11;
    if ( v11 )
    {
      v13 = sub_8D3110(v18);
      v15 = sub_8D46C0(v18);
      v12 = 0;
      a4 = v17;
      a1 = v15;
      if ( (*(_BYTE *)(v15 + 140) & 0xFB) == 8 )
      {
        v16 = sub_8D4C10(v15, unk_4F077C4 != 2);
        a4 = v17;
        v12 = v16;
      }
    }
    if ( a2 != 2 )
    {
      if ( a2 != 5 )
      {
        if ( a2 == 1 )
          return sub_697AE0(i, v12, v13, a5, a6, 0, (__int64)v20);
LABEL_14:
        sub_721090(a1);
      }
      return sub_697B80(i, v12, v13, a4, a5, a6, (__int64)v20);
    }
    return *(_QWORD *)(v19 + 24);
  }
  if ( a2 == 2 )
    return *(_QWORD *)(v19 + 24);
  v13 = 0;
  v12 = 0;
  if ( a2 == 5 )
    return sub_697B80(i, v12, v13, a4, a5, a6, (__int64)v20);
  if ( a2 != 1 )
    goto LABEL_14;
  return sub_697930(a1, 1, 1, 0, a5, a6, 0, 0);
}
