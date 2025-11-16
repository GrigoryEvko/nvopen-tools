// Function: sub_68CED0
// Address: 0x68ced0
//
__int64 __fastcall sub_68CED0(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  _QWORD *v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  int v10; // ebx
  __int16 v11; // r15
  __int64 v12; // rax
  int v13; // eax
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a1;
  v20 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v21[0] = *(_QWORD *)&dword_4F063F8;
  v8 = HIDWORD(qword_4F077B4);
  if ( !HIDWORD(qword_4F077B4) )
  {
    if ( (unsigned int)sub_6E5430() )
    {
      a2 = (unsigned int *)v21;
      a1 = 1103;
      sub_6851C0(0x44Fu, v21);
    }
    goto LABEL_22;
  }
  if ( (*(_BYTE *)(unk_4D03C50 + 19LL) & 0x40) != 0 )
  {
    v15 = *(_BYTE *)(unk_4D03C50 + 16LL);
    if ( v15 )
    {
      if ( v15 == 1 )
      {
        if ( (unsigned int)sub_6E5430() )
        {
          a2 = (unsigned int *)v21;
          a1 = 60;
          sub_6851C0(0x3Cu, v21);
        }
      }
      else
      {
        if ( v15 != 2 )
          goto LABEL_5;
        if ( (unsigned int)sub_6E5430() )
        {
          a2 = (unsigned int *)v21;
          a1 = 529;
          sub_6851C0(0x211u, v21);
        }
      }
    }
    else if ( (unsigned int)sub_6E5430() )
    {
      a2 = (unsigned int *)v21;
      a1 = 58;
      sub_6851C0(0x3Au, v21);
    }
LABEL_22:
    sub_7B8B50(a1, a2, v16, v17);
    v10 = qword_4F063F0;
    v11 = WORD2(qword_4F063F0);
    if ( dword_4F04C58 != -1 )
    {
      sub_64E550(0, 0);
LABEL_19:
      sub_6E6260(v6);
      goto LABEL_10;
    }
LABEL_16:
    if ( (unsigned int)sub_6E5430() )
    {
      a2 = &dword_4F063F8;
      a1 = 1226;
      sub_6851C0(0x4CAu, &dword_4F063F8);
    }
    sub_7B8B50(a1, a2, v18, v19);
    goto LABEL_19;
  }
  if ( unk_4D04320 )
  {
    a2 = &dword_4F063F8;
    a1 = 1103;
    sub_684B30(0x44Fu, &dword_4F063F8);
  }
LABEL_5:
  if ( dword_4F04C58 != -1 )
  {
    v7 = qword_4F04C68;
    v9 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
    if ( v9 )
    {
      if ( (*(_BYTE *)(v9 - 8) & 0x10) == 0 && (*(_BYTE *)(v9 + 198) & 0x10) != 0 )
      {
        if ( (unsigned int)sub_6E5430() )
        {
          a2 = (unsigned int *)v21;
          a1 = 3539;
          sub_6851C0(0xDD3u, v21);
        }
        goto LABEL_22;
      }
    }
  }
  sub_7B8B50(a1, a2, v8, v7);
  v10 = qword_4F063F0;
  v11 = WORD2(qword_4F063F0);
  if ( dword_4F04C58 == -1 )
    goto LABEL_16;
  v12 = sub_64E550(0, 0);
  sub_72D4C0(v12, v20);
  sub_6E6A50(v20, v6);
LABEL_10:
  v13 = v21[0];
  *(_DWORD *)(v6 + 76) = v10;
  *(_WORD *)(v6 + 80) = v11;
  *(_DWORD *)(v6 + 68) = v13;
  LOWORD(v13) = WORD2(v21[0]);
  *(_BYTE *)(v6 + 17) = 2;
  *(_WORD *)(v6 + 72) = v13;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v6 + 68);
  unk_4F061D8 = *(_QWORD *)(v6 + 76);
  sub_6E3280(v6, v21);
  sub_6E3BA0(v6, v21, 0, 0);
  sub_6E26D0(1, v6);
  return sub_724E30(&v20);
}
