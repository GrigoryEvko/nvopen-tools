// Function: sub_9286A0
// Address: 0x9286a0
//
__int64 __fastcall sub_9286A0(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int64 a8,
        unsigned __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  __int64 v16; // rdi
  _QWORD *v17; // rax
  _QWORD *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rbx
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 v24; // rax
  __int64 v25; // rdx

  if ( (_DWORD)a7 )
  {
    if ( (_DWORD)a7 == 1 )
    {
      sub_925930(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
    }
    else
    {
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
    }
  }
  else
  {
    v16 = *(_QWORD *)(a2 + 32);
    v17 = *(_QWORD **)(v16 + 520);
    if ( !v17 )
      goto LABEL_9;
    v18 = (_QWORD *)(v16 + 512);
    do
    {
      while ( 1 )
      {
        a4 = v17[2];
        v19 = v17[3];
        if ( v17[4] >= a8 )
          break;
        v17 = (_QWORD *)v17[3];
        if ( !v19 )
          goto LABEL_7;
      }
      v18 = v17;
      v17 = (_QWORD *)v17[2];
    }
    while ( a4 );
LABEL_7:
    if ( (_QWORD *)(v16 + 512) != v18 && v18[4] <= a8 )
    {
      v24 = sub_928510(a2, a8, (__int64)a3, a4);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v24;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
    }
    else
    {
LABEL_9:
      v20 = sub_91A390(v16 + 8, a9, 0, a4);
      v21 = *(unsigned __int8 *)(v20 + 8);
      if ( (unsigned __int8)v21 <= 3u || (_BYTE)v21 == 5 )
        goto LABEL_10;
      if ( (unsigned __int8)v21 > 0x14u )
      {
        if ( *(_BYTE *)(*(_QWORD *)(a8 + 8) + 8LL) != 14 )
          goto LABEL_19;
        goto LABEL_25;
      }
      v25 = 1463376;
      if ( _bittest64(&v25, v21) )
      {
LABEL_10:
        v22 = sub_926480(a2, a8, v20, (unsigned int)a10, a13 & 1);
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = v22;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)(a8 + 8) + 8LL) != 14 )
LABEL_19:
          sub_91B8A0("unexpected error generating l-value!", a3, 1);
        if ( (_BYTE)v21 != 13 )
LABEL_25:
          sub_91B8A0("unexpected error generating l-value!", a3, 1);
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = a8;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
      }
    }
  }
  return a1;
}
