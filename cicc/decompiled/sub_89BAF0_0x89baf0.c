// Function: sub_89BAF0
// Address: 0x89baf0
//
_BOOL8 __fastcall sub_89BAF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int16 a7,
        int a8,
        int a9,
        unsigned int a10,
        int a11)
{
  __int16 v15; // r12
  __int64 v16; // rax
  _BOOL4 v17; // eax
  __int64 v18; // rcx
  _BOOL8 v19; // r8
  __int64 v20; // rbx
  __int64 v21; // rax
  _UNKNOWN *__ptr32 *v22; // r8
  __int64 v23; // rsi
  __int16 v24; // ax
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // [rsp+8h] [rbp-48h]
  _BOOL4 v31; // [rsp+8h] [rbp-48h]
  _BOOL4 v33; // [rsp+10h] [rbp-40h]

  v15 = a7;
  if ( !sub_88FB10(a1, a2) )
  {
    v30 = sub_892920(a2);
    v16 = sub_892920(a1);
    v17 = sub_88FB10(v16, v30);
    v19 = v17;
    if ( !v17 )
    {
      if ( *(_BYTE *)(a1 + 80) != 19 )
        return v19;
      v26 = *(_QWORD *)(a1 + 88);
      if ( (*(_BYTE *)(v26 + 160) & 2) != 0 )
      {
        if ( *(_BYTE *)(a2 + 80) != 19 )
          return v19;
        v27 = *(_QWORD *)(a2 + 88);
        if ( (*(_BYTE *)(v27 + 160) & 2) == 0 || *(_QWORD *)a1 != *(_QWORD *)a2 )
        {
          if ( (*(_BYTE *)(v26 + 266) & 1) == 0 )
            return v19;
LABEL_28:
          if ( (*(_BYTE *)(v27 + 266) & 1) == 0 )
            return v19;
          v33 = v19;
          if ( !(unsigned int)sub_89B9E0(v26, v27, a9 != 0, 0) )
          {
            LODWORD(v19) = v33;
            return v19;
          }
          goto LABEL_3;
        }
        v31 = v17;
        v28 = (unsigned __int8)(a9 != 0) << 6;
        v29 = (a9 != 0) << 6;
        BYTE1(v29) |= 1u;
        if ( a10 )
          v28 = v29;
        if ( (unsigned int)sub_8D97D0(
                             *(_QWORD *)(*(_QWORD *)(a3 + 40) + 32LL),
                             *(_QWORD *)(*(_QWORD *)(a4 + 40) + 32LL),
                             v28,
                             v18,
                             v19) )
          goto LABEL_3;
        LODWORD(v19) = v31;
        if ( *(_BYTE *)(a1 + 80) != 19 )
          return v19;
        v26 = *(_QWORD *)(a1 + 88);
      }
      if ( (*(_BYTE *)(v26 + 266) & 1) == 0 || *(_BYTE *)(a2 + 80) != 19 )
        return v19;
      v27 = *(_QWORD *)(a2 + 88);
      goto LABEL_28;
    }
  }
LABEL_3:
  v20 = sub_892920(a1);
  v21 = sub_892920(a2);
  v23 = *(_QWORD *)(v21 + 88);
  if ( (*(_BYTE *)(*(_QWORD *)(v20 + 88) + 160LL) & 8) != 0 || (*(_BYTE *)(v23 + 160) & 8) != 0 )
    v15 = a7 | 0x20;
  if ( a8 )
    v15 |= 1u;
  if ( *(_BYTE *)(v20 + 80) == 19 && (*(_BYTE *)(*(_QWORD *)(v20 + 88) + 160LL) & 2) != 0 )
  {
    v15 |= 2u;
  }
  else if ( *(_BYTE *)(v21 + 80) == 19 && (*(_BYTE *)(v23 + 160) & 2) != 0 )
  {
    v15 |= 2u;
  }
  if ( a9 )
    v15 |= 0x10u;
  if ( a10 )
    v15 |= 0x40u;
  HIBYTE(v24) = HIBYTE(v15);
  if ( !a11 )
  {
    LOBYTE(v24) = v15 | 0x80;
    v15 = v24;
  }
  LODWORD(v19) = sub_89AB40(a5, a6, v15, a10, v22);
  return v19;
}
