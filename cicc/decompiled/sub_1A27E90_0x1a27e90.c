// Function: sub_1A27E90
// Address: 0x1a27e90
//
_QWORD *__fastcall sub_1A27E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v8; // rdx
  __int64 v9; // r14
  unsigned int v10; // ebx
  _QWORD *v11; // rdx
  _QWORD *result; // rax
  _QWORD *v13; // rbx
  __int64 v14; // rax
  unsigned int v15; // ebx
  unsigned __int64 v16; // rbx
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // r14
  int v21; // eax
  unsigned __int8 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // eax
  __int64 v26; // rdx
  int v27; // edi
  unsigned int v28; // esi
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  char v32; // al
  int v33; // eax
  __int64 v34; // rax
  _QWORD *v35; // rdx
  int v36; // eax
  unsigned int v37; // [rsp+8h] [rbp-78h]
  unsigned __int64 v38; // [rsp+8h] [rbp-78h]
  unsigned __int64 v39; // [rsp+8h] [rbp-78h]
  unsigned __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+10h] [rbp-70h] BYREF
  int v42; // [rsp+18h] [rbp-68h] BYREF
  char v43[16]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v44; // [rsp+30h] [rbp-50h]
  char v45; // [rsp+40h] [rbp-40h]

  v8 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v9 = *(_QWORD *)(a2 + 24 * (2 - v8));
  if ( *(_BYTE *)(v9 + 16) == 13 )
  {
    v10 = *(_DWORD *)(v9 + 32);
    if ( v10 > 0x40 )
    {
      if ( v10 - (unsigned int)sub_16A57B0(v9 + 24) <= 0x40 && !**(_QWORD **)(v9 + 24) )
        return (_QWORD *)sub_1A21B40(a1, a2, v8, a4, a5, a6);
    }
    else if ( !*(_QWORD *)(v9 + 24) )
    {
      return (_QWORD *)sub_1A21B40(a1, a2, v8, a4, a5, a6);
    }
  }
  else
  {
    v9 = 0;
  }
  v11 = *(_QWORD **)(a1 + 560);
  result = *(_QWORD **)(a1 + 552);
  if ( v11 == result )
  {
    v13 = &result[*(unsigned int *)(a1 + 572)];
    if ( result == v13 )
    {
      v35 = *(_QWORD **)(a1 + 552);
    }
    else
    {
      do
      {
        if ( a2 == *result )
          break;
        ++result;
      }
      while ( v13 != result );
      v35 = v13;
    }
  }
  else
  {
    v13 = &v11[*(unsigned int *)(a1 + 568)];
    result = sub_16CC9F0(a1 + 544, a2);
    if ( a2 == *result )
    {
      v23 = *(_QWORD *)(a1 + 560);
      if ( v23 == *(_QWORD *)(a1 + 552) )
        v24 = *(unsigned int *)(a1 + 572);
      else
        v24 = *(unsigned int *)(a1 + 568);
      v35 = (_QWORD *)(v23 + 8 * v24);
    }
    else
    {
      v14 = *(_QWORD *)(a1 + 560);
      if ( v14 != *(_QWORD *)(a1 + 552) )
      {
        result = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a1 + 568));
        goto LABEL_8;
      }
      result = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a1 + 572));
      v35 = result;
    }
  }
  while ( v35 != result && *result >= 0xFFFFFFFFFFFFFFFELL )
    ++result;
LABEL_8:
  if ( result == v13 )
  {
    if ( !*(_BYTE *)(a1 + 344) )
    {
      *(_QWORD *)(a1 + 8) = *(_QWORD *)(a1 + 8) & 3LL | a2 | 4;
      return result;
    }
    v15 = *(_DWORD *)(a1 + 360);
    a6 = *(_QWORD *)(a1 + 368);
    if ( v15 > 0x40 )
    {
      v39 = *(_QWORD *)(a1 + 368);
      v25 = sub_16A57B0(a1 + 352);
      a6 = v39;
      if ( v15 - v25 <= 0x40 )
      {
        v16 = **(_QWORD **)(a1 + 352);
        if ( v39 > v16 )
          goto LABEL_16;
      }
    }
    else
    {
      v16 = *(_QWORD *)(a1 + 352);
      if ( a6 > v16 )
      {
LABEL_16:
        v17 = a6 - v16;
        if ( v9 )
        {
          v37 = *(_DWORD *)(v9 + 32);
          if ( v37 > 0x40 )
          {
            v33 = sub_16A57B0(v9 + 24);
            v17 = -1;
            if ( v37 - v33 <= 0x40 )
              v17 = **(_QWORD **)(v9 + 24);
          }
          else
          {
            v17 = *(_QWORD *)(v9 + 24);
          }
        }
        v18 = **(_QWORD **)(a1 + 336);
        if ( *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) == v18
          && v18 == *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) )
        {
          v40 = v17;
          v32 = sub_1A211D0(a2);
          a6 = v40;
          if ( !v32 )
            return (_QWORD *)sub_1A21B40(a1, a2, v8, a4, a5, a6);
          goto LABEL_24;
        }
        v19 = *(_QWORD *)(a1 + 376);
        v38 = v17;
        v41 = a2;
        v42 = *(_DWORD *)(v19 + 16);
        sub_1A27C40((__int64)v43, a1 + 384, &v41, &v42);
        a6 = v38;
        if ( v45 )
        {
          v22 = v9 != 0;
          return (_QWORD *)sub_1A22CF0((_QWORD *)a1, a2, a1 + 352, a6, v22, a6);
        }
        v20 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 376) + 8LL) + 24LL * *(unsigned int *)(v44 + 8));
        v21 = sub_1A211D0(a2);
        a6 = v38;
        LODWORD(a5) = v21;
        if ( (_BYTE)v21 || *v20 != v16 )
        {
          v20[2] &= ~4uLL;
LABEL_24:
          v22 = 0;
          return (_QWORD *)sub_1A22CF0((_QWORD *)a1, a2, a1 + 352, a6, v22, a6);
        }
        v20[2] &= 7uLL;
        return (_QWORD *)sub_1A21B40(a1, a2, v8, a4, a5, a6);
      }
    }
    a4 = *(_BYTE *)(a1 + 392) & 1;
    if ( (*(_BYTE *)(a1 + 392) & 1) != 0 )
    {
      v26 = a1 + 400;
      v27 = 3;
    }
    else
    {
      v31 = *(unsigned int *)(a1 + 408);
      v26 = *(_QWORD *)(a1 + 400);
      if ( !(_DWORD)v31 )
        goto LABEL_60;
      v27 = v31 - 1;
    }
    v28 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v29 = v26 + 16LL * (v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)));
    a5 = *(_QWORD *)v29;
    if ( a2 == *(_QWORD *)v29 )
    {
LABEL_43:
      v30 = 64;
      if ( !(_BYTE)a4 )
        v30 = 16LL * *(unsigned int *)(a1 + 408);
      v8 = v30 + v26;
      if ( v29 != v8 )
      {
        a4 = *(_QWORD *)(a1 + 376);
        v8 = 3LL * *(unsigned int *)(v29 + 8);
        *(_QWORD *)(*(_QWORD *)(a4 + 8) + 24LL * *(unsigned int *)(v29 + 8) + 16) &= 7uLL;
      }
      return (_QWORD *)sub_1A21B40(a1, a2, v8, a4, a5, a6);
    }
    v36 = 1;
    while ( a5 != -8 )
    {
      LODWORD(a6) = v36 + 1;
      v28 = v27 & (v36 + v28);
      v29 = v26 + 16LL * v28;
      a5 = *(_QWORD *)v29;
      if ( a2 == *(_QWORD *)v29 )
        goto LABEL_43;
      v36 = a6;
    }
    if ( (_BYTE)a4 )
    {
      v34 = 64;
      goto LABEL_61;
    }
    v31 = *(unsigned int *)(a1 + 408);
LABEL_60:
    v34 = 16 * v31;
LABEL_61:
    v29 = v26 + v34;
    goto LABEL_43;
  }
  return result;
}
