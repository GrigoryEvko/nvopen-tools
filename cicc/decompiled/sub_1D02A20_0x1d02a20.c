// Function: sub_1D02A20
// Address: 0x1d02a20
//
__int64 __fastcall sub_1D02A20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // ecx
  unsigned __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned int *v13; // rax
  unsigned int v15; // r15d
  unsigned int v16; // r13d
  char v17; // al
  unsigned int v18; // r9d
  unsigned __int16 v19; // cx
  int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // r8
  unsigned int v23; // edi
  _WORD *v24; // r11
  unsigned __int16 v25; // dx
  __int16 *v26; // rdi
  _WORD *v27; // r11
  int v28; // eax
  unsigned __int16 *v29; // r8
  unsigned int v30; // r11d
  unsigned int i; // esi
  __int16 *v32; // r11
  __int16 v33; // di
  int v34; // esi
  __int64 v35; // [rsp+0h] [rbp-60h]
  unsigned int v36; // [rsp+Ch] [rbp-54h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  unsigned __int8 v40; // [rsp+2Bh] [rbp-35h]
  int v41; // [rsp+2Ch] [rbp-34h]

  v7 = *(_QWORD *)(a3 + 8) + ((unsigned __int64)(unsigned int)~*(__int16 *)(a1 + 24) << 6);
  v36 = *(unsigned __int8 *)(v7 + 4);
  v37 = *(_QWORD *)(v7 + 32);
  while ( 1 )
  {
    if ( !a2 )
      return 0;
    v8 = *(_QWORD *)(a2 + 32);
    v9 = *(_DWORD *)(a2 + 56);
    if ( *(__int16 *)(a2 + 24) < 0 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)(a3 + 8) + ((__int64)~*(__int16 *)(a2 + 24) << 6) + 32);
      v11 = v8 + 40LL * v9;
      if ( v11 == v8 )
      {
LABEL_13:
        v39 = 0;
        if ( !v10 )
          goto LABEL_8;
      }
      else
      {
        v12 = *(_QWORD *)(a2 + 32);
        while ( *(_WORD *)(*(_QWORD *)v12 + 24LL) != 9 )
        {
          v12 += 40;
          if ( v11 == v12 )
            goto LABEL_13;
        }
        v39 = *(_QWORD *)(*(_QWORD *)v12 + 88LL);
        if ( !(v10 | v39) )
          goto LABEL_8;
      }
      v41 = *(_DWORD *)(a1 + 60);
      if ( v36 != v41 )
        break;
    }
LABEL_8:
    if ( v9 )
    {
      v13 = (unsigned int *)(v8 + 40LL * (v9 - 1));
      a2 = *(_QWORD *)v13;
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v13 + 40LL) + 16LL * v13[2]) == 111 )
        continue;
    }
    return 0;
  }
  v35 = a2;
  v15 = v36;
  v16 = 0;
  while ( 1 )
  {
    v17 = *(_BYTE *)(*(_QWORD *)(a1 + 40) + 16LL * v15);
    if ( v17 != 1 && v17 != 111 )
    {
      v40 = sub_1D18C40(a1);
      if ( v40 )
      {
        v18 = *(unsigned __int16 *)(v37 + 2LL * v16);
        v19 = *(_WORD *)(v37 + 2LL * v16);
        if ( v39 && ((*(_DWORD *)(v39 + 4LL * (v18 >> 5)) >> v19) & 1) == 0 )
          return 1;
        if ( v10 )
        {
          v20 = *(unsigned __int16 *)v10;
          if ( (_WORD)v20 )
            break;
        }
      }
    }
LABEL_33:
    ++v15;
    ++v16;
    if ( v41 == v15 )
    {
      v8 = *(_QWORD *)(v35 + 32);
      v9 = *(_DWORD *)(v35 + 56);
      goto LABEL_8;
    }
  }
LABEL_23:
  if ( v18 != (unsigned __int16)v20 )
  {
    v21 = *(_QWORD *)(a4 + 8);
    v22 = *(_QWORD *)(a4 + 56);
    v23 = *(_DWORD *)(v21 + 24LL * v19 + 16);
    v24 = (_WORD *)(v22 + 2LL * (v23 >> 4));
    v25 = *v24 + v19 * (v23 & 0xF);
    v26 = v24 + 1;
    LODWORD(v24) = *(_DWORD *)(v21 + 24LL * (unsigned __int16)v20 + 16);
    v28 = ((unsigned __int8)v24 & 0xF) * v20;
    v27 = (_WORD *)(v22 + 2LL * ((unsigned int)v24 >> 4));
    LOWORD(v28) = *v27 + v28;
    v29 = v27 + 1;
    v30 = v25;
    for ( i = (unsigned __int16)v28; v30 != i; i = (unsigned __int16)v28 )
    {
      if ( v30 < i )
      {
        while ( 1 )
        {
          v32 = v26 + 1;
          v33 = *v26;
          v25 += v33;
          if ( !v33 )
            break;
          v26 = v32;
          v30 = v25;
          if ( v25 == i )
            return v40;
          if ( v25 >= i )
            goto LABEL_30;
        }
LABEL_32:
        v20 = *(unsigned __int16 *)(v10 + 2);
        v10 += 2LL;
        if ( (_WORD)v20 )
          goto LABEL_23;
        goto LABEL_33;
      }
LABEL_30:
      v34 = *v29;
      if ( !(_WORD)v34 )
        goto LABEL_32;
      v28 += v34;
      ++v29;
    }
  }
  return v40;
}
