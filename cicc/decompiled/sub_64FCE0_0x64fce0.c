// Function: sub_64FCE0
// Address: 0x64fce0
//
__int64 __fastcall sub_64FCE0(__int64 *a1)
{
  _QWORD *v1; // r13
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  _BYTE *v5; // rax
  __int64 v6; // rax
  unsigned int v7; // r12d
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v11; // rax
  char v12; // dl
  char v13; // al

  v1 = a1 + 6;
  v2 = *a1;
  if ( !*((_DWORD *)a1 + 12) )
    v1 = a1 + 3;
  if ( !v2 )
  {
    v11 = a1[36];
    if ( !v11 )
      goto LABEL_18;
    while ( 1 )
    {
      v12 = *(_BYTE *)(v11 + 140);
      if ( v12 != 12 )
        break;
      v11 = *(_QWORD *)(v11 + 160);
    }
    if ( v12 )
LABEL_18:
      sub_6851C0((a1[1] & 0x20) == 0 ? 64 : 2840, v1);
    goto LABEL_11;
  }
  if ( *(_BYTE *)(v2 + 80) != 7 )
  {
    sub_6851C0(2840, v1);
LABEL_11:
    v6 = sub_87EBB0(7, 0);
    *a1 = v6;
    *(_BYTE *)(v6 + 81) |= 0x20u;
    v7 = dword_4F04C5C;
    v8 = sub_72C930(7);
    v3 = sub_735FB0(v8, 3, v7, v9);
    sub_877D80(v3, *a1);
    *(_BYTE *)(v3 + 177) = 1;
    *(_QWORD *)(v3 + 184) = sub_72C9A0();
    *(_QWORD *)(*a1 + 88) = v3;
    goto LABEL_12;
  }
  if ( *((char *)a1 + 121) < 0 )
  {
    sub_6851C0(2843, v1);
    goto LABEL_11;
  }
  v3 = *(_QWORD *)(v2 + 88);
  if ( (*((_BYTE *)a1 + 131) & 0x10) != 0 )
  {
    v4 = *(_QWORD *)(v3 + 128);
    if ( v4 )
    {
      v5 = *(_BYTE **)(v4 + 16);
      *(_BYTE *)(*(_QWORD *)v5 + 81LL) |= 1u;
      v5[169] |= 0x10u;
    }
    sub_6851C0(2950, v1);
    goto LABEL_11;
  }
  if ( dword_4F077BC | (unsigned int)qword_4F077B4 | dword_4D04964 )
  {
    if ( *((_BYTE *)a1 + 268) )
      goto LABEL_21;
  }
  else if ( *(_BYTE *)(v3 + 136) != 3 )
  {
LABEL_21:
    sub_6851C0(80, a1 + 4);
    goto LABEL_11;
  }
  if ( (a1[1] & 0x401000) != 0 )
    goto LABEL_21;
  if ( (unsigned int)sub_8D3410(*(_QWORD *)(v3 + 120)) )
  {
    sub_6851C0(701, a1 + 3);
    goto LABEL_11;
  }
  if ( (a1[1] & 0x20) != 0 )
    sub_6851C0(255, a1 + 4);
  v13 = *(_BYTE *)(v3 + 174);
  if ( (v13 & 0x10) == 0 )
  {
    sub_6851C0(2841, v1);
    goto LABEL_11;
  }
  if ( (v13 & 0x20) != 0 )
    sub_6851C0(2842, v1);
LABEL_12:
  sub_8756B0(*a1);
  return v3;
}
