// Function: sub_2A114C0
// Address: 0x2a114c0
//
_QWORD *__fastcall sub_2A114C0(__int64 a1, unsigned int a2)
{
  __int64 v3; // rdi
  unsigned __int8 v4; // dl
  __int64 *v5; // r8
  char *v6; // r8
  char v7; // r9
  char v8; // cl
  int v9; // ecx
  __int64 v11; // r8
  unsigned int v12; // r9d
  unsigned int v13; // r8d
  int v14; // r10d
  _QWORD *v15; // rdi
  _BYTE *v16; // rax
  unsigned int v17; // eax
  unsigned int v18; // edi
  unsigned int v19; // edx
  int v20; // ecx
  unsigned int v21; // eax
  char v22; // cl
  _QWORD *v23; // rdi
  unsigned int v24; // edx
  unsigned int v25; // r9d
  unsigned int v26; // eax
  __int64 v27; // [rsp+18h] [rbp-28h]
  __int64 v28; // [rsp+20h] [rbp-20h]

  v3 = a1 - 16;
  v4 = *(_BYTE *)(a1 - 16);
  if ( (v4 & 2) == 0 )
  {
    v11 = v3 - 8LL * ((v4 >> 2) & 0xF);
    if ( **(_BYTE **)v11 == 20 )
    {
      if ( (*(_DWORD *)(*(_QWORD *)v11 + 4LL) & 7) == 7 && (*(_DWORD *)(*(_QWORD *)v11 + 4LL) & 0xFFFFFFF8) != 0 )
        return (_QWORD *)a1;
      v6 = *(char **)(v3 - 8LL * ((v4 >> 2) & 0xF));
      v7 = *v6;
      goto LABEL_18;
    }
LABEL_6:
    v8 = qword_4F813A8[8];
LABEL_7:
    if ( v8 )
      goto LABEL_12;
    v9 = 0;
    goto LABEL_9;
  }
  v5 = *(__int64 **)(a1 - 32);
  if ( *(_BYTE *)*v5 != 20 )
    goto LABEL_6;
  if ( (*(_DWORD *)(*v5 + 4) & 7) == 7 && (*(_DWORD *)(*v5 + 4) & 0xFFFFFFF8) != 0 )
    return (_QWORD *)a1;
  v6 = (char *)*v5;
  v7 = *v6;
LABEL_18:
  v8 = qword_4F813A8[8];
  if ( v7 != 20 )
    goto LABEL_7;
  v12 = *((_DWORD *)v6 + 1);
  if ( !LOBYTE(qword_4F813A8[8]) )
  {
    v13 = v12 >> 1;
    v14 = (v12 & 2) != 0;
    if ( (v12 & 1) == 0 )
    {
      if ( (v12 & 0x40) != 0 )
        v25 = v12 >> 14;
      else
        v25 = v12 >> 7;
      v13 = v25;
      v14 = v25 & 1;
    }
    if ( !v14 )
    {
      v9 = (v13 >> 1) & 0x1F;
      if ( ((v13 >> 1) & 0x20) != 0 )
        v9 |= (v13 >> 2) & 0xFE0;
LABEL_9:
      if ( !v9 )
        v9 = 1;
      a2 *= v9;
      v8 = 0;
    }
  }
LABEL_12:
  if ( a2 <= 1 )
    return (_QWORD *)a1;
  if ( (*(_BYTE *)(a1 - 16) & 2) == 0 )
  {
    v15 = (_QWORD *)(v3 - 8LL * ((v4 >> 2) & 0xF));
    v16 = (_BYTE *)*v15;
    if ( *(_BYTE *)*v15 == 20 )
      goto LABEL_26;
LABEL_34:
    v19 = 0;
    v18 = 0;
    v22 = 7;
    goto LABEL_35;
  }
  v23 = *(_QWORD **)(a1 - 32);
  v16 = (_BYTE *)*v23;
  if ( *(_BYTE *)*v23 != 20 )
    goto LABEL_34;
LABEL_26:
  v17 = *((_DWORD *)v16 + 1);
  if ( (v17 & 7) == 7 && (v17 & 0xFFFFFFF8) != 0 )
  {
    if ( (v17 & 0x10000000) != 0 )
      v18 = HIWORD(v17) & 7;
    else
      v18 = (unsigned __int16)(v17 >> 3);
  }
  else
  {
    v18 = (unsigned __int8)v17;
    if ( !v8 )
    {
      v18 = 0;
      if ( (v17 & 1) == 0 )
      {
        v18 = (v17 >> 1) & 0x1F;
        if ( ((v17 >> 1) & 0x20) != 0 )
          v18 |= (v17 >> 2) & 0xFE0;
      }
    }
  }
  v19 = v17 >> 1;
  v20 = (v17 & 2) != 0;
  if ( (v17 & 1) == 0 )
  {
    if ( (v17 & 0x40) != 0 )
      v26 = v17 >> 14;
    else
      v26 = v17 >> 7;
    v19 = v26;
    v20 = v26 & 1;
  }
  v21 = v19 >> 1;
  if ( v20 )
    goto LABEL_36;
  v22 = (v19 & 0x40) == 0 ? 7 : 14;
LABEL_35:
  v21 = v19 >> v22;
LABEL_36:
  v24 = 0;
  if ( (v21 & 1) == 0 )
  {
    v24 = (v21 >> 1) & 0x1F;
    if ( ((v21 >> 1) & 0x20) != 0 )
      v24 |= (v21 >> 2) & 0xFE0;
  }
  v27 = sub_AF17B0(v18, a2, v24);
  if ( BYTE4(v27) )
    return sub_26BDBC0(a1, v27);
  else
    return (_QWORD *)v28;
}
