// Function: sub_34A1780
// Address: 0x34a1780
//
__int64 __fastcall sub_34A1780(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        char a10,
        __int64 a11)
{
  __int64 v12; // r12
  int v13; // ebx
  int v14; // eax
  __int64 *v15; // rdx
  __int64 v16; // rdi
  bool v17; // al
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v21; // rbx
  unsigned int v22; // [rsp+14h] [rbp-9Ch]
  __int64 *v23; // [rsp+20h] [rbp-90h]
  unsigned int v24; // [rsp+28h] [rbp-88h]
  int v25; // [rsp+3Ch] [rbp-74h] BYREF
  __int64 v26; // [rsp+40h] [rbp-70h] BYREF
  __int64 v27; // [rsp+48h] [rbp-68h] BYREF
  __int64 v28; // [rsp+50h] [rbp-60h] BYREF
  char v29; // [rsp+68h] [rbp-48h]
  __int64 v30; // [rsp+70h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 824) & 1) != 0 )
  {
    v12 = a2 + 832;
    v13 = 7;
  }
  else
  {
    v12 = *(_QWORD *)(a2 + 832);
    v21 = *(unsigned int *)(a2 + 840);
    v18 = v12;
    if ( !(_DWORD)v21 )
      goto LABEL_26;
    v13 = v21 - 1;
  }
  v29 = 0;
  v28 = 0;
  v30 = 0;
  v25 = 0;
  if ( a10 )
    v25 = (unsigned __int16)a9 | ((_DWORD)a8 << 16);
  v27 = a11;
  v26 = a7;
  v14 = sub_F11290(&v26, &v25, &v27);
  v15 = &v28;
  v22 = 1;
  v24 = v13 & v14;
  while ( 1 )
  {
    v16 = v12 + 72LL * v24;
    if ( a7 == *(_QWORD *)v16
      && a10 == *(_BYTE *)(v16 + 24)
      && (!a10 || a8 == *(_QWORD *)(v16 + 8) && a9 == *(_QWORD *)(v16 + 16))
      && a11 == *(_QWORD *)(v16 + 32) )
    {
      if ( (*(_BYTE *)(a2 + 824) & 1) != 0 )
        goto LABEL_10;
      v18 = *(_QWORD *)(a2 + 832);
      goto LABEL_24;
    }
    v23 = v15;
    v17 = sub_F34140(v16, (__int64)v15);
    v15 = v23;
    if ( v17 )
      break;
    a4 = v22;
    v24 = v13 & (v22 + v24);
    ++v22;
  }
  if ( (*(_BYTE *)(a2 + 824) & 1) != 0 )
  {
    v16 = a2 + 1408;
LABEL_10:
    v18 = a2 + 832;
    v19 = 576;
    goto LABEL_11;
  }
  v18 = *(_QWORD *)(a2 + 832);
  v21 = *(unsigned int *)(a2 + 840);
LABEL_26:
  v16 = v18 + 72 * v21;
LABEL_24:
  v19 = 72LL * *(unsigned int *)(a2 + 840);
LABEL_11:
  if ( v16 == v19 + v18 )
  {
    *(_BYTE *)(a1 + 32) = 0;
  }
  else
  {
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x200000000LL;
    if ( *(_DWORD *)(v16 + 48) )
      sub_349DD80(a1, v16 + 40, v19, a4, a5, a6);
    *(_BYTE *)(a1 + 32) = 1;
  }
  return a1;
}
