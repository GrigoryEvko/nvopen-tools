// Function: sub_1F47EC0
// Address: 0x1f47ec0
//
__int64 __fastcall sub_1F47EC0(__int64 *a1)
{
  int v2; // ecx
  __int64 v3; // rax
  int v4; // eax
  __int64 v5; // rax
  __int64 (*v6)(); // rax
  unsigned int v7; // r13d
  __int64 (*v9)(); // rax
  __int64 v10; // rax
  void (*v11)(); // rdx
  __int64 (*v12)(); // rax
  __int64 v13; // rax
  void (*v14)(); // rdx
  __int64 (*v15)(); // rax
  __int64 v16; // rax
  void (*v17)(); // rdx
  __int64 (*v18)(); // rax
  __int64 v19; // rsi
  bool (*v20)(); // rax
  unsigned __int8 v21; // di
  _QWORD *v22; // rsi

  v2 = dword_4FCC660;
  *(_BYTE *)(a1[26] + 640) = (2 * (dword_4FCC660 != 2)) | *(_BYTE *)(a1[26] + 640) & 0xFD;
  if ( v2 == 1 )
  {
    v3 = a1[26];
LABEL_3:
    *(_BYTE *)(v3 + 800) |= 2u;
    goto LABEL_4;
  }
  if ( !(unsigned int)sub_1700720(a1[26]) )
  {
    v3 = a1[26];
    if ( (*(_BYTE *)(v3 + 640) & 2) == 0 )
    {
      v4 = dword_4FCC580;
      if ( dword_4FCC580 != 1 )
        goto LABEL_5;
      goto LABEL_14;
    }
    goto LABEL_3;
  }
LABEL_4:
  v4 = dword_4FCC580;
  if ( dword_4FCC580 != 1 )
  {
LABEL_5:
    if ( v4 )
      goto LABEL_8;
    v5 = a1[26];
    if ( (*(_BYTE *)(v5 + 800) & 4) == 0 || dword_4FCC660 == 1 )
      goto LABEL_8;
    goto LABEL_15;
  }
LABEL_14:
  v5 = a1[26];
LABEL_15:
  *(_BYTE *)(v5 + 800) &= ~2u;
  v9 = *(__int64 (**)())(*a1 + 192);
  if ( v9 == sub_1F445C0 || ((unsigned __int8 (__fastcall *)(__int64 *))v9)(a1) )
    return 1;
  v10 = *a1;
  v11 = *(void (**)())(*a1 + 200);
  if ( v11 != nullsub_761 )
  {
    ((void (__fastcall *)(__int64 *))v11)(a1);
    v10 = *a1;
  }
  v12 = *(__int64 (**)())(v10 + 208);
  if ( v12 == sub_1F445E0 || ((unsigned __int8 (__fastcall *)(__int64 *))v12)(a1) )
    return 1;
  v13 = *a1;
  v14 = *(void (**)())(*a1 + 216);
  if ( v14 != nullsub_762 )
  {
    ((void (__fastcall *)(__int64 *))v14)(a1);
    v13 = *a1;
  }
  v15 = *(__int64 (**)())(v13 + 224);
  if ( v15 == sub_1F44600 || ((unsigned __int8 (__fastcall *)(__int64 *))v15)(a1) )
    return 1;
  v16 = *a1;
  v17 = *(void (**)())(*a1 + 232);
  if ( v17 != nullsub_763 )
  {
    ((void (__fastcall *)(__int64 *))v17)(a1);
    v16 = *a1;
  }
  v18 = *(__int64 (**)())(v16 + 240);
  if ( v18 == sub_1F44620 )
    return 1;
  v7 = ((__int64 (__fastcall *)(__int64 *))v18)(a1);
  if ( (_BYTE)v7 )
    return 1;
  v19 = sub_1F47DD0((__int64)a1);
  v20 = *(bool (**)())(*a1 + 272);
  if ( v20 == sub_1F448F0 )
  {
    v21 = dword_4FCC380 == 2;
  }
  else
  {
    v19 = (unsigned int)v19;
    v21 = ((__int64 (__fastcall *)(__int64 *))v20)(a1);
  }
  v22 = (_QWORD *)sub_210E000(v21, v19);
  sub_1F46490((__int64)a1, v22, 1, 1, 1u);
  if ( sub_1F47DD0((__int64)a1) )
    return v7;
LABEL_8:
  v6 = *(__int64 (**)())(*a1 + 184);
  if ( v6 == sub_1F445B0 )
    return 1;
  return ((__int64 (__fastcall *)(__int64 *))v6)(a1);
}
