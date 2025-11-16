// Function: sub_651D50
// Address: 0x651d50
//
__int64 __fastcall sub_651D50(__int64 *a1, __int64 **a2, __int64 a3, __int64 a4)
{
  int v4; // r15d
  __int64 v5; // r13
  __int64 **v6; // r12
  int v7; // ebx
  _QWORD *v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  unsigned int v15; // r12d
  __int64 v16; // rax
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rax
  int v25; // ebx
  char v26; // al
  int v27; // [rsp+18h] [rbp-288h] BYREF
  int v28; // [rsp+1Ch] [rbp-284h]
  __int64 v29; // [rsp+20h] [rbp-280h] BYREF
  __int64 v30; // [rsp+28h] [rbp-278h] BYREF
  _QWORD v31[12]; // [rsp+30h] [rbp-270h] BYREF
  _QWORD v32[66]; // [rsp+90h] [rbp-210h] BYREF

  v4 = 0;
  v5 = a3;
  v6 = a2;
  v7 = (int)a1;
  v30 = *(_QWORD *)&dword_4F063F8;
  if ( (_DWORD)a1 )
  {
    if ( a3 )
    {
LABEL_3:
      v8 = qword_4F04C68;
      v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      LOBYTE(v8) = v7 != 0;
      *(_QWORD *)(v9 + 680) = v5;
      LOBYTE(a4) = v6 != 0;
      v7 = 0;
      a4 = 4 * ((unsigned int)a4 & (unsigned int)v8);
      a3 = (unsigned int)a4 | *(_BYTE *)(v9 + 11) & 0xFB;
      *(_BYTE *)(v9 + 11) = a4 | *(_BYTE *)(v9 + 11) & 0xFB;
      goto LABEL_4;
    }
    a1 = *a2;
    if ( *a2
      && (v4 = sub_8D23B0(a1)) == 0
      && (a1 = *a2, (v24 = **a2) != 0)
      && (v25 = *(unsigned __int8 *)(v24 + 80), a3 = (unsigned int)(v25 - 4), (unsigned __int8)(v25 - 4) <= 1u)
      && ((v26 = *(_BYTE *)(*(_QWORD *)(v24 + 88) + 177LL), (v26 & 0x20) == 0) || v26 < 0) )
    {
      a2 = 0;
      v7 = 1;
      sub_8646E0(a1, 0);
    }
    else
    {
      v4 = 0;
      v7 = 0;
    }
  }
  else
  {
    if ( a2 )
    {
      a1 = *a2;
      if ( *a2 )
      {
        a2 = 0;
        v4 = 1;
        sub_864360(a1, 0);
      }
    }
    if ( v5 )
      goto LABEL_3;
  }
LABEL_4:
  sub_7B8B50(a1, a2, a3, a4);
  v10 = 0;
  v11 = 0;
  if ( (unsigned int)sub_651570(0, 0, 0x10u) )
  {
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    memset(v32, 0, 0x1D8u);
    v29 = *(_QWORD *)&dword_4F063F8;
    v32[3] = *(_QWORD *)&dword_4F063F8;
    v32[19] = v32;
    if ( !dword_4F077BC )
      goto LABEL_21;
    goto LABEL_12;
  }
  v15 = dword_4F077BC;
  if ( dword_4F077BC )
  {
    v15 = 0;
    if ( word_4F06418[0] == 142 )
    {
      *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
      memset(v32, 0, 0x1D8u);
      v29 = *(_QWORD *)&dword_4F063F8;
      v32[3] = *(_QWORD *)&dword_4F063F8;
      v32[19] = v32;
LABEL_12:
      if ( qword_4F077A8 <= 0x9F5Fu )
        BYTE2(v32[22]) |= 1u;
LABEL_21:
      BYTE2(v32[15]) |= 0x40u;
      BYTE4(v32[16]) |= 0x80u;
      memset(v31, 0, 0x58u);
      v18 = (-(__int64)(dword_4D043F8 == 0) & 0xFFFFFFFFF8000000LL) + 151519234;
      if ( dword_4D043E0 )
        v18 = ((-(__int64)(dword_4D043F8 == 0) & 0xFFFFFFFFF8000000LL) + 151519234) | 0x400000;
      if ( unk_4F0774C )
        BYTE4(v32[15]) |= 0x40u;
      sub_672A20(v18, v32, v31);
      if ( (v32[1] & 0x20) != 0 )
      {
        sub_6851C0(255, &v29);
      }
      else if ( (v32[1] & 1) == 0 )
      {
        sub_64E990((__int64)dword_4F07508, v32[34], 0, 0, 0, 1);
      }
      v19 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      *(_BYTE *)(v19 + 11) &= ~4u;
      v20 = v32[34];
      *(_QWORD *)(v19 + 680) = 0;
      v21 = sub_626600(v20, (__int64)v32, 1, 0, 0, 0, 0, &v27, (__int64)v31);
      v22 = v21;
      if ( (v32[15] & 0x8000000000LL) != 0 )
      {
        v32[35] = v21;
        v32[36] = v21;
        sub_625720((__int64)v32);
        if ( (v32[15] & 0x8000000000LL) == 0 )
          v22 = sub_72C930(v32);
      }
      if ( qword_4D0495C && (v23 = v22, (unsigned int)sub_6454D0(v22, (__int64)&v29))
        || v27 && (v23 = v22, (unsigned int)sub_8DD040(v22, &v29)) )
      {
        v22 = sub_72C930(v23);
      }
      sub_7BEC40();
      word_4F06418[0] = 1;
      *(_QWORD *)dword_4F07508 = v30;
      *(_QWORD *)&dword_4F063F8 = v30;
      if ( LODWORD(v31[7]) )
      {
        qword_4F063F0 = v31[7];
        if ( (*(_BYTE *)(v22 + 140) & 0xFB) != 8 )
          goto LABEL_34;
      }
      else
      {
        qword_4F063F0 = v31[5];
        if ( (*(_BYTE *)(v22 + 140) & 0xFB) != 8 )
          goto LABEL_34;
      }
      if ( (unsigned int)sub_8D4C10(v22, dword_4F077C4 != 2) )
      {
        v28 = 0;
        sub_6243B0(v22, (__int64)v32, (__int64)dword_4F07508);
        if ( v28 )
          v22 = sub_72C930(v22);
      }
LABEL_34:
      if ( (v32[15] & 0x2000000000LL) != 0 )
        sub_6451E0((__int64)v32);
      sub_643EB0((__int64)v32, 0);
      v10 = &qword_4D04A00;
      v11 = v22;
      v15 = 1;
      sub_87AB50(v11, &qword_4D04A00, &v30);
      if ( v7 )
        goto LABEL_37;
      goto LABEL_7;
    }
  }
  if ( v7 )
  {
LABEL_37:
    sub_866010(v11, v10, v12, v13, v14);
    return v15;
  }
LABEL_7:
  if ( v4 )
  {
    sub_8645D0();
  }
  else
  {
    v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    *(_BYTE *)(v16 + 11) &= ~4u;
    *(_QWORD *)(v16 + 680) = 0;
  }
  return v15;
}
