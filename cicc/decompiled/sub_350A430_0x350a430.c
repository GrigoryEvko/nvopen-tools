// Function: sub_350A430
// Address: 0x350a430
//
__int64 __fastcall sub_350A430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v14; // rsi
  unsigned __int64 v15; // rax
  int v16; // r9d
  __int64 i; // r10
  __int16 v18; // dx
  __int64 v19; // r10
  __int64 v20; // rdi
  unsigned int v21; // esi
  __int64 *v22; // rdx
  __int64 v23; // r11
  __int64 v24; // r15
  __int64 v25; // rdi
  __int64 (__fastcall *v26)(__int64, __int64); // rax
  __int64 v27; // rax
  int v29; // edx
  int v30; // [rsp+Ch] [rbp-34h]

  if ( *(_BYTE *)(a1 + 108) )
  {
    v10 = *(_QWORD **)(a1 + 88);
    v11 = &v10[*(unsigned int *)(a1 + 100)];
    if ( v10 == v11 )
      return 0;
    while ( a3 != *v10 )
    {
      if ( v11 == ++v10 )
        return 0;
    }
LABEL_6:
    v12 = *(_QWORD *)(a2 + 8);
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
    v14 = v12;
    v15 = v12;
    if ( (*(_DWORD *)(v12 + 44) & 4) != 0 )
    {
      do
        v15 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v15 + 44) & 4) != 0 );
    }
    v16 = *(_DWORD *)(v12 + 44) & 8;
    if ( v16 )
    {
      do
        v14 = *(_QWORD *)(v14 + 8);
      while ( (*(_BYTE *)(v14 + 44) & 8) != 0 );
    }
    for ( i = *(_QWORD *)(v14 + 8); i != v15; v15 = *(_QWORD *)(v15 + 8) )
    {
      v18 = *(_WORD *)(v15 + 68);
      if ( (unsigned __int16)(v18 - 14) > 4u && v18 != 24 )
        break;
    }
    v19 = *(_QWORD *)(v13 + 128);
    v20 = *(unsigned int *)(v13 + 144);
    if ( (_DWORD)v20 )
    {
      v21 = (v20 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v22 = (__int64 *)(v19 + 16LL * v21);
      v23 = *v22;
      if ( v15 == *v22 )
        goto LABEL_16;
      v29 = 1;
      while ( v23 != -4096 )
      {
        v21 = (v20 - 1) & (v29 + v21);
        v30 = v29 + 1;
        v22 = (__int64 *)(v19 + 16LL * v21);
        v23 = *v22;
        if ( *v22 == v15 )
          goto LABEL_16;
        v29 = v30;
      }
    }
    v22 = (__int64 *)(v19 + 16 * v20);
LABEL_16:
    v24 = v22[1];
    if ( a5 )
    {
      v25 = *(_QWORD *)(a1 + 48);
      v26 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v25 + 176LL);
      if ( v26 == sub_2E4F5F0 )
      {
        if ( (*(_DWORD *)(v12 + 44) & 4) != 0 || !v16 )
          v27 = (*(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL) >> 30) & 1LL;
        else
          LOBYTE(v27) = sub_2E88A90(v12, 0x40000000, 2);
      }
      else
      {
        LOBYTE(v27) = v26(v25, v12);
      }
      if ( !(_BYTE)v27 )
        return 0;
      v12 = *(_QWORD *)(a2 + 8);
    }
    return sub_3509FC0((__int64 *)a1, v12, v24, a4);
  }
  if ( sub_C8CA60(a1 + 80, a3) )
    goto LABEL_6;
  return 0;
}
