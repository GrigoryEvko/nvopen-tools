// Function: sub_646C60
// Address: 0x646c60
//
__int64 __fastcall sub_646C60(__int64 a1)
{
  unsigned int v2; // r15d
  __int64 i; // r12
  __int64 v4; // rdi
  char v5; // al
  __int64 v6; // r13
  unsigned int v7; // r15d
  __int64 v8; // rdi
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v19; // rax
  __int64 v20; // rax

  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned int)sub_8D32B0(*(_QWORD *)(a1 + 288))
      && (v19 = sub_8D46C0(*(_QWORD *)(a1 + 288)), (unsigned int)sub_8D2310(v19))
      || (unsigned int)sub_8D3D10(*(_QWORD *)(a1 + 288))
      && (v20 = sub_8D4870(*(_QWORD *)(a1 + 288)), (unsigned int)sub_8D2310(v20)) )
    {
      sub_6464A0(*(_QWORD *)(a1 + 288), *(_QWORD *)a1, (unsigned int *)(a1 + 48), 1u);
    }
  }
  v2 = sub_8DED30(*(_QWORD *)(a1 + 288), *(_QWORD *)(a1 + 296), HIDWORD(qword_4F077B4) == 0 ? 5 : 13);
  if ( v2 )
    goto LABEL_21;
  for ( i = *(_QWORD *)(a1 + 296); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v4 = *(_QWORD *)(a1 + 288);
  v5 = *(_BYTE *)(v4 + 140);
  v6 = *(_QWORD *)(a1 + 288);
  if ( v5 == 12 )
  {
    do
      v6 = *(_QWORD *)(v6 + 160);
    while ( *(_BYTE *)(v6 + 140) == 12 );
  }
  if ( dword_4F077C0 && qword_4F077A8 <= 0x752Fu )
  {
    if ( v6 == i )
    {
LABEL_12:
      v7 = 0;
      if ( (v5 & 0xFB) == 8 )
        v7 = sub_8D4C10(v4, dword_4F077C4 != 2);
      v8 = *(_QWORD *)(a1 + 296);
      if ( (*(_BYTE *)(v8 + 140) & 0xFB) == 8 )
        v7 |= sub_8D4C10(v8, dword_4F077C4 != 2);
      v9 = v6;
      v10 = 0;
      v11 = sub_73C570(v9, v7, -1);
      *(_QWORD *)(a1 + 288) = v11;
      if ( (*(_BYTE *)(v11 + 140) & 0xFB) == 8 )
        v10 = sub_8D4C10(v11, dword_4F077C4 != 2);
      v12 = *(_QWORD *)(a1 + 296);
      if ( (*(_BYTE *)(v12 + 140) & 0xFB) == 8 )
        v10 |= sub_8D4C10(v12, dword_4F077C4 != 2);
      v13 = sub_73C570(i, v10, -1);
      v14 = *(_QWORD *)a1;
      *(_QWORD *)(a1 + 296) = v13;
      sub_6853B0(5, 147, a1 + 48, v14);
LABEL_21:
      v2 = 1;
      v15 = sub_8D79B0(*(_QWORD *)(a1 + 288), *(_QWORD *)(a1 + 296));
      v16 = *(_QWORD *)a1;
      *(_QWORD *)(a1 + 288) = v15;
      *(_QWORD *)(*(_QWORD *)(v16 + 88) + 120LL) = v15;
      return v2;
    }
    if ( (unsigned int)sub_8DED30(v6, i, 5) )
    {
      v4 = *(_QWORD *)(a1 + 288);
      v5 = *(_BYTE *)(v4 + 140);
      goto LABEL_12;
    }
  }
  v17 = *(_QWORD *)a1;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 81LL) & 2) != 0 )
    *(_QWORD *)(a1 + 288) = *(_QWORD *)(a1 + 296);
  sub_6853B0(8, 147, a1 + 48, v17);
  return v2;
}
