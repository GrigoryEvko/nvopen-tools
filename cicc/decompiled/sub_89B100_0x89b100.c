// Function: sub_89B100
// Address: 0x89b100
//
_QWORD *__fastcall sub_89B100(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // al
  __int64 v5; // rbx
  __int64 v7; // rax
  _UNKNOWN *__ptr32 *v8; // r8
  __int64 v9; // rdx
  __int64 v10; // rcx
  const __m128i *v11; // r15
  __int64 v12; // rax
  char v13; // al
  __int16 v14; // bx
  _QWORD *v15; // r14
  __int16 v16; // bx
  char v17; // al
  __int64 v18; // rdi
  char v19; // al
  __int64 v20; // rax
  __m128i **v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // [rsp+0h] [rbp-40h]
  __int64 v24; // [rsp+8h] [rbp-38h]

  v4 = *(_BYTE *)(a1 + 80);
  if ( v4 == 9 || v4 == 7 )
  {
    v5 = *(_QWORD *)(a1 + 88);
    if ( (*(_BYTE *)(v5 + 170) & 0x40) == 0 )
    {
LABEL_5:
      sub_6854C0(0x342u, (FILE *)(a3 + 8), a1);
      *(_DWORD *)(a2 + 52) = 1;
      *(_BYTE *)(a3 + 17) |= 0x20u;
      *(_QWORD *)(a3 + 24) = 0;
      return sub_898DA0(a2, a3, 0);
    }
  }
  else
  {
    if ( v4 != 21 )
      BUG();
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 192LL);
    if ( (*(_BYTE *)(v5 + 170) & 0x40) == 0 )
      goto LABEL_5;
  }
  *(_DWORD *)(a2 + 28) = 1;
  v7 = sub_892920(**(_QWORD **)(*(_QWORD *)(v5 + 216) + 16LL));
  v9 = *(_QWORD *)(v7 + 88);
  v10 = v7;
  v23 = v7;
  v24 = v9;
  v11 = **(const __m128i ***)(v5 + 216);
  v12 = v9;
  if ( *(_BYTE *)(v10 + 80) == 19 )
  {
    v22 = *(_QWORD *)(v9 + 200);
    if ( v22 )
      v12 = *(_QWORD *)(v22 + 88);
  }
  v13 = *(_BYTE *)(v12 + 160);
  v14 = 2 * ((v13 & 6) != 0);
  if ( (v13 & 0x10) != 0 )
    v14 = (2 * ((v13 & 6) != 0)) | 0x20;
  v15 = *(_QWORD **)(v24 + 144);
  if ( v15 )
  {
    v16 = v14 | 8;
    while ( 1 )
    {
      v17 = *((_BYTE *)v15 + 80);
      if ( v17 == 19 )
      {
        v18 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v15[11] + 176LL) + 88LL) + 168LL) + 168LL);
      }
      else
      {
        if ( v17 != 21 )
          sub_721090();
        v18 = **(_QWORD **)(*(_QWORD *)(v15[11] + 192LL) + 216LL);
      }
      if ( sub_89AB40(v18, (__int64)v11, v16, v10, v8)
        && (*((_BYTE *)v15 + 80) != 19 || sub_890680(*(_QWORD *)(v15[11] + 176LL), 0)) )
      {
        break;
      }
      v15 = (_QWORD *)v15[1];
      if ( !v15 )
        goto LABEL_20;
    }
  }
  else
  {
LABEL_20:
    *(_BYTE *)(*(_QWORD *)a2 + 127LL) |= 0x10u;
    v15 = sub_898DA0(a2, a3, v23);
    *(_QWORD *)(v15[11] + 152LL) = v23;
    v19 = *((_BYTE *)v15 + 80);
    if ( v19 == 9 || v19 == 7 )
    {
      v20 = v15[11];
    }
    else
    {
      if ( v19 != 21 )
        BUG();
      v20 = *(_QWORD *)(v15[11] + 192LL);
    }
    v21 = *(__m128i ***)(v20 + 216);
    if ( (*(_BYTE *)(v23 + 81) & 0x10) != 0 && !*(_QWORD *)(a2 + 240) )
      sub_890490(a2, *(__int64 **)(v23 + 64));
    if ( *(_DWORD *)(a2 + 64) || *(_DWORD *)(a2 + 72) )
    {
      *((_BYTE *)v15 + 81) |= 0x10u;
      v15[8] = *(_QWORD *)(v23 + 64);
    }
    v21[1] = *v21;
    *v21 = sub_72F240(v11);
    if ( !*(_DWORD *)(a2 + 52) && (*(_BYTE *)(a3 + 17) & 0x20) == 0 )
    {
      v15[1] = *(_QWORD *)(v24 + 144);
      *(_QWORD *)(v24 + 144) = v15;
    }
  }
  return v15;
}
