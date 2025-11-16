// Function: sub_5ED880
// Address: 0x5ed880
//
void __fastcall sub_5ED880(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  char v6; // cl
  __int64 v7; // rax
  char v8; // dl
  __int64 i; // r12
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rdi
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v6 = *(_BYTE *)(a2 + 140);
  if ( v6 == 12 )
  {
    v7 = a2;
    do
    {
      v7 = *(_QWORD *)(v7 + 160);
      v8 = *(_BYTE *)(v7 + 140);
    }
    while ( v8 == 12 );
  }
  else
  {
    v8 = *(_BYTE *)(a2 + 140);
  }
  if ( v8 && (unk_4F07590 || (*(_BYTE *)(a1 + 177) & 0x20) == 0) )
  {
    for ( i = a2; v6 == 12; v6 = *(_BYTE *)(i + 140) )
      i = *(_QWORD *)(i + 160);
    if ( v6 == 14 )
      i = sub_7CFE40(i);
    if ( a1 == i
      && ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 || !(unsigned int)sub_8D3E20(a1)) )
    {
      sub_684B30(522, dword_4F07508);
    }
    else
    {
      v10 = *(_QWORD *)(i + 168);
      if ( !a3 )
      {
        v11 = *(_QWORD **)(v10 + 128);
        if ( v11 )
        {
          while ( a1 != v11[1] )
          {
            v11 = (_QWORD *)*v11;
            if ( !v11 )
              goto LABEL_17;
          }
          v19 = *(_QWORD *)(i + 168);
          sub_684B00(324, dword_4F07508);
          v10 = v19;
        }
      }
LABEL_17:
      v20 = v10;
      v12 = (_QWORD *)sub_727D20(i);
      v12[1] = a1;
      *v12 = *(_QWORD *)(v20 + 128);
      *(_QWORD *)(v20 + 128) = v12;
      v21 = *(_QWORD *)(a1 + 168);
      v13 = (_QWORD *)sub_727D20(a1);
      v13[1] = i;
      *v13 = *(_QWORD *)(v21 + 144);
      *(_QWORD *)(v21 + 144) = v13;
    }
    if ( !(unk_4F04C3C | a3) )
    {
      v14 = sub_8D21F0(a2);
      v15 = sub_86A2A0(v14);
      if ( v15 && *(_BYTE *)(v15 + 16) == 53 )
      {
        v17 = *(_QWORD *)(v15 + 24);
      }
      else
      {
        v16 = sub_86A1D0(a2, 6, a2);
        *(_BYTE *)(v16 + 57) |= 1u;
        v17 = v16;
        v18 = *(_BYTE *)(v16 - 8) & 1;
        *(_QWORD *)v16 = *(_QWORD *)dword_4F07508;
        *(_QWORD *)(v16 + 8) = sub_729420(v18, a4);
        if ( !unk_4F04C3C )
          sub_8699D0(v17, 53, 0);
      }
      *(_BYTE *)(v17 + 57) |= 4u;
      *(_QWORD *)(v17 + 32) = a2;
    }
  }
}
