// Function: sub_7B78D0
// Address: 0x7b78d0
//
__int64 __fastcall sub_7B78D0(unsigned int a1)
{
  unsigned int v1; // r12d
  const char *v2; // r8
  int v3; // eax
  _BYTE *v4; // r10
  int v5; // r15d
  unsigned int v6; // r12d
  unsigned int v7; // esi
  unsigned int v8; // r12d
  const char *v9; // rcx
  __int64 *v10; // rax
  _BYTE *v11; // rsi
  unsigned __int64 v12; // rdx
  int v13; // esi
  unsigned int v14; // edx
  unsigned int v15; // r13d
  __int64 *v17; // r9
  const char *v18; // rsi
  int v19; // ebx
  __int64 v20; // r13
  const char *v21; // r11
  int v22; // edx
  const char *v23; // rcx
  const char *v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdx
  int v27; // r13d
  const char **v28; // r12
  const char *v29; // rdi
  __int64 **v30; // r13
  _BYTE *v31; // rax
  _BYTE *v32; // rax
  int v33; // [rsp+8h] [rbp-218h]
  __int64 v35; // [rsp+8h] [rbp-218h]
  const char *v36; // [rsp+10h] [rbp-210h]
  _BYTE *v37; // [rsp+10h] [rbp-210h]
  __int64 v38; // [rsp+18h] [rbp-208h]
  __int64 v39; // [rsp+20h] [rbp-200h] BYREF
  const char *v40; // [rsp+28h] [rbp-1F8h] BYREF
  const char *v41; // [rsp+30h] [rbp-1F0h] BYREF
  _BYTE *v42; // [rsp+38h] [rbp-1E8h] BYREF
  _QWORD v43[2]; // [rsp+40h] [rbp-1E0h] BYREF
  _QWORD v44[2]; // [rsp+50h] [rbp-1D0h] BYREF
  _QWORD v45[2]; // [rsp+60h] [rbp-1C0h] BYREF
  const char *v46[16]; // [rsp+70h] [rbp-1B0h] BYREF
  unsigned __int64 v47[38]; // [rsp+F0h] [rbp-130h] BYREF

  v1 = a1;
  v39 = 0;
  v40 = 0;
  v2 = qword_4F06460;
  v38 = unk_4D03BE0;
  v43[0] = unk_4D03BE0;
  v43[1] = &v40;
  v44[0] = v43;
  v44[1] = &v41;
  v45[0] = v44;
  v45[1] = &v42;
  unk_4D03BE0 = v45;
  v3 = (a1 & 8) != 0;
  v41 = qword_4F06460;
  if ( (a1 & 7) <= 2 )
  {
    if ( (a1 & 7) == 2 )
      v3 += 2;
  }
  else
  {
    ++v3;
  }
  v4 = &qword_4F06460[~(__int64)v3];
  v42 = v4;
  if ( (a1 & 8) == 0 )
  {
    v2 = 0;
    v5 = -1;
    goto LABEL_5;
  }
  v40 = qword_4F06460;
  v17 = (__int64 *)unk_4F06458;
  if ( unk_4F06458 )
  {
    do
    {
      if ( (unsigned __int64)qword_4F06460 <= v17[1] )
        break;
      v17 = (__int64 *)*v17;
    }
    while ( v17 );
  }
  v18 = qword_4F06460;
  v19 = 0;
  v20 = 0;
  v33 = 0;
  while ( 1 )
  {
    v21 = v18;
    if ( v18 > &v2[16 - v20] )
      break;
    v22 = *v18;
    if ( *v18 == 40 )
    {
      v23 = v18 + 1;
LABEL_51:
      v27 = unk_4D03CE4 == 0 ? 7 : 4;
      if ( v19 )
      {
        v36 = v23;
        v28 = v46;
        do
        {
          v29 = *v28++;
          sub_7B0EB0((unsigned __int64)v29, (__int64)dword_4F07508);
          sub_684AC0(v27, 0x992u);
        }
        while ( &v46[(unsigned int)(v19 - 1) + 1] != v28 );
        v23 = v36;
        v1 = a1;
        v2 = v40;
        v4 = v42;
      }
      qword_4F06460 = v23;
      v41 = v23;
      v5 = (_DWORD)v23 - (_DWORD)v2 - 1;
      goto LABEL_5;
    }
    if ( !(_BYTE)v22 )
      break;
    v23 = ++v18;
    if ( v17 && (const char *)v17[1] == v21 )
    {
      if ( *((_DWORD *)v17 + 4) )
      {
        v22 = 92;
        goto LABEL_45;
      }
      if ( *((_BYTE *)v17 + 24) == 40 )
        goto LABEL_51;
      v33 += 2;
      v17 = (__int64 *)*v17;
      v20 = v33;
    }
    else if ( !dword_4F04DC0[(char)v22 + 128] )
    {
LABEL_45:
      if ( v19 )
      {
        v24 = v46[v19 - 1];
        if ( *v24 || v21 != v24 + 1 )
          goto LABEL_48;
      }
      else
      {
        memset(v47, 0, 0x100u);
LABEL_48:
        v25 = v22 + 128;
        if ( !*((_BYTE *)v47 + v25) )
        {
          *((_BYTE *)v47 + v25) = 1;
          v26 = v19++;
          v46[v26] = v21;
        }
      }
    }
  }
  sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
  sub_684AC0(8u, 0x993u);
  if ( unk_4D03D20 )
    ++qword_4F06410;
  v4 = v42;
  v1 = a1 & 0xFFFFFFF7;
  v2 = v40;
  v5 = -1;
LABEL_5:
  if ( (unsigned int)sub_7B6B00(&v39, 0, v1, 34, v2, v5, v4, 0) )
  {
    if ( unk_4D042B8 )
      goto LABEL_7;
    if ( v5 < 0 && (!HIDWORD(qword_4F077B4) || qword_4F077A8 > 0x765Bu) )
      goto LABEL_32;
    v37 = v42;
    v35 = unk_4D03BE0;
    v47[0] = unk_4D03BE0;
    v47[1] = (unsigned __int64)v46;
    unk_4D03BE0 = v47;
    v46[0] = v40;
    while ( !*qword_4F06460 )
    {
      if ( qword_4F06460[1] != 2 )
        break;
      v30 = (__int64 **)qword_4F06450;
      *(_DWORD *)(sub_7ABAA0(2, (__int64)qword_4F06460) + 24) = unk_4F06468 + 1;
      v31 = qword_4F06460++;
      *v31 = 92;
      v32 = qword_4F06460++;
      *v32 = 110;
      if ( sub_7B2B10(0, 1) )
        break;
      qword_4F06460 -= 2;
      if ( !(unsigned int)sub_7B6B00(&v39, 0, v1, 34, v46[0], v5, v37, v30) )
      {
        unk_4D03BE0 = v35;
        goto LABEL_14;
      }
    }
    unk_4D03BE0 = v35;
LABEL_7:
    if ( v5 >= 0 )
    {
      unk_4F06208 = 2452;
      if ( !unk_4D03D20 )
        goto LABEL_9;
LABEL_33:
      v8 = 0;
LABEL_16:
      v9 = v40;
      if ( v40 )
      {
        v10 = (__int64 *)unk_4F06458;
        if ( unk_4F06458 )
        {
          v11 = qword_4F06460;
          do
          {
            v12 = v10[1];
            if ( v12 >= (unsigned __int64)v11 )
              break;
            if ( (unsigned __int64)v9 <= v12 )
              *((_BYTE *)v10 + 20) = 1;
            v10 = (__int64 *)*v10;
          }
          while ( v10 );
        }
      }
    }
    else
    {
LABEL_32:
      unk_4F06208 = 8;
      if ( unk_4D03D20 )
        goto LABEL_33;
LABEL_9:
      sub_72C970((__int64)xmmword_4F06300);
      v6 = unk_4F06208;
      sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
      v7 = v6;
      v8 = 7;
      sub_684AC0(8u, v7);
    }
  }
  else
  {
LABEL_14:
    ++qword_4F06460;
    if ( unk_4D03D20 )
    {
      v8 = 7;
      goto LABEL_16;
    }
    v13 = v5;
    if ( v5 >= 0 && v40[v5] == 91 )
      v13 = v5 + 2;
    v14 = v1;
    v8 = 7;
    sub_7CE2C0((_DWORD)v41, qword_4F06408 + ~v13, v14, v39, (unsigned int)v46, (unsigned int)v47, 0);
    v15 = (unsigned int)v46[0];
    if ( LODWORD(v46[0]) )
    {
      sub_7B0EB0(v47[0], (__int64)dword_4F07508);
      sub_684AC0(8u, v15);
    }
  }
  unk_4D03BE0 = v38;
  return v8;
}
