// Function: sub_1341860
// Address: 0x1341860
//
unsigned __int64 __fastcall sub_1341860(__int64 a1, __int64 a2, __int64 *a3, int a4, int a5, char a6, char a7)
{
  int v7; // r10d
  char v12; // r11
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r12
  _QWORD *v15; // rdx
  unsigned __int64 v16; // rcx
  unsigned __int64 *v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rax
  unsigned __int64 *v21; // rdx
  __int64 v22; // rcx
  unsigned __int64 v23; // r12
  __int64 v24; // rax
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // r8
  _QWORD *v28; // r9
  unsigned int i; // r8d
  _QWORD *v30; // r9
  _QWORD *v31; // rdx
  unsigned __int64 v32; // rax
  int v34; // [rsp+Ch] [rbp-1C4h]
  char v35; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v36; // [rsp+18h] [rbp-1B8h]
  _QWORD v37[54]; // [rsp+20h] [rbp-1B0h] BYREF

  v7 = a5;
  v12 = a7;
  v13 = a3[1] & 0xFFFFFFFFFFFFF000LL;
  v14 = v13 - 4096;
  if ( a6 )
    v14 = v13 + (a3[2] & 0xFFFFFFFFFFFFF000LL);
  if ( !v14 )
    return 0;
  v15 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    sub_130D500(v37);
    v15 = v37;
    v12 = a7;
    v7 = a5;
  }
  v16 = v14 & 0xFFFFFFFFC0000000LL;
  v17 = (_QWORD *)((char *)v15 + ((v14 >> 26) & 0xF0));
  v18 = *v17;
  if ( *v17 == (v14 & 0xFFFFFFFFC0000000LL) )
  {
    v19 = (unsigned __int64 *)(v17[1] + ((v14 >> 9) & 0x1FFFF8));
  }
  else if ( v16 == v15[32] )
  {
    v27 = v15[33];
    v15[32] = v18;
    v19 = (unsigned __int64 *)(v27 + ((v14 >> 9) & 0x1FFFF8));
    v15[33] = v17[1];
    *v17 = v16;
    v17[1] = v27;
  }
  else
  {
    v28 = v15 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( v16 == *v28 )
      {
        v30 = &v15[2 * i];
        v36 = v30[33];
        v31 = &v15[2 * i - 2];
        v30[32] = v31[32];
        v30[33] = v31[33];
        v31[32] = v18;
        v31[33] = v17[1];
        *v17 = v16;
        v17[1] = v36;
        v19 = (unsigned __int64 *)(v36 + ((v14 >> 9) & 0x1FFFF8));
        goto LABEL_8;
      }
      v28 += 2;
    }
    v34 = v7;
    v35 = v12;
    v32 = sub_130D370(a1, a2, v15, v14, 0, 0);
    v7 = v34;
    v12 = v35;
    v19 = (unsigned __int64 *)v32;
  }
LABEL_8:
  if ( !v19 )
    return 0;
  v20 = *v19;
  v21 = (unsigned __int64 *)(((__int64)(*v19 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL);
  if ( !v21 )
    return 0;
  v22 = *a3;
  if ( !a6 )
  {
    if ( (v22 & 0x100000000000LL) != 0 )
      return 0;
    v23 = ((__int64)(*v19 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL;
    v24 = (v20 >> 2) & 7;
    if ( !a4 )
      goto LABEL_13;
LABEL_18:
    if ( !v24 )
      return 0;
    v26 = *v21;
    goto LABEL_20;
  }
  if ( (v20 & 2) != 0 )
    return 0;
  v23 = ((__int64)(*v19 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL;
  v24 = (v20 >> 2) & 7;
  if ( a4 )
    goto LABEL_18;
LABEL_13:
  if ( v7 != (_DWORD)v24 )
    return 0;
  v26 = *v21;
  if ( !v12 && ((*v21 & 0x2000) != 0) != ((*a3 & 0x2000) != 0) )
    return 0;
LABEL_20:
  if ( a4 == ((v26 >> 14) & 1) && (unk_4C6F2C8 || (v26 & 0xFFF) == (v22 & 0xFFF)) )
  {
    sub_1341570(a1, a2, v21, 5u);
    return v23;
  }
  return 0;
}
