// Function: sub_66A7C0
// Address: 0x66a7c0
//
void __fastcall sub_66A7C0(__int64 a1, int a2, __int64 a3, int a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned __int8 v9; // al
  __int64 v10; // rbx
  char v11; // dl

  v4 = *(_QWORD *)(a1 + 88);
  if ( dword_4F04C58 != -1 )
  {
    v5 = *(_QWORD *)(unk_4F04C50 + 32LL);
    *(_BYTE *)(v5 + 202) |= 0x20u;
LABEL_3:
    *(_QWORD *)(v4 + 48) = v5;
    v6 = 1;
    goto LABEL_4;
  }
  v6 = unk_4F04C38;
  if ( unk_4F04C38 )
  {
    v7 = qword_4F04C68[0] + 776LL * a4;
    if ( *(_DWORD *)(v7 + 400) == -1 )
    {
      v11 = *(_BYTE *)(v7 + 4);
      if ( (unsigned __int8)(v11 - 6) > 1u )
      {
        while ( v11 )
        {
          if ( v11 == 17 )
            goto LABEL_8;
          v11 = *(_BYTE *)(v7 - 772);
          v7 -= 776;
        }
        v5 = 0;
      }
      else
      {
        v5 = *(_QWORD *)(*(_QWORD *)(v7 + 208) + 48LL);
      }
    }
    else
    {
LABEL_8:
      v5 = *(_QWORD *)(v7 + 216);
    }
    goto LABEL_3;
  }
LABEL_4:
  if ( dword_4F077C4 == 2 )
  {
    v8 = qword_4F04C68[0] + 776LL * a4;
    v9 = *(_BYTE *)(v8 + 4);
    if ( v9 == 6 )
    {
      if ( a2 )
      {
        sub_877E20(a1, v4, *(_QWORD *)(v8 + 208));
        *(_BYTE *)(v4 + 88) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 5) & 3 | *(_BYTE *)(v4 + 88) & 0xFC;
      }
    }
    else if ( v9 > 6u )
    {
      if ( v9 == 7 )
      {
        v10 = *(_QWORD *)(v4 + 168);
        sub_877E20(a1, v4, *(_QWORD *)(v8 + 208));
        if ( (*(_BYTE *)(v10 + 109) & 0x20) != 0 )
          *(_BYTE *)(v4 + 88) &= 0xFCu;
      }
    }
    else if ( (unsigned __int8)(v9 - 3) <= 1u )
    {
      sub_877E90(a1, v4);
    }
    if ( !v6 )
      sub_66A6A0(v4);
  }
}
