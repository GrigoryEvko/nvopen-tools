// Function: sub_1039CC0
// Address: 0x1039cc0
//
__int64 __fastcall sub_1039CC0(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        _BYTE *a7)
{
  _BYTE *v9; // r12
  unsigned int v10; // eax
  unsigned __int64 v11; // rax
  bool v12; // al
  _BYTE *v13; // r13
  int v14; // ebx
  __int64 v15; // r12
  __int64 v16; // r14
  unsigned int v17; // ebx
  __int64 v19; // rax
  _BYTE *v20; // rsi
  __int64 v21; // rax
  _BYTE *v22; // rsi
  __int64 v23; // [rsp+18h] [rbp-78h]
  _BOOL4 v24; // [rsp+20h] [rbp-70h]
  __int64 v27; // [rsp+38h] [rbp-58h] BYREF
  __int64 v28; // [rsp+40h] [rbp-50h] BYREF
  __int64 v29; // [rsp+48h] [rbp-48h]
  __int64 v30; // [rsp+50h] [rbp-40h]

  v9 = a2;
  LOBYTE(v10) = sub_10394B0(*a2);
  if ( !(_BYTE)v10 )
  {
    v11 = *((_QWORD *)a2 + 9);
    if ( !v11 )
      goto LABEL_11;
    v12 = v11 > 1;
    if ( a2 + 40 != *((unsigned __int8 **)a2 + 7) )
    {
      v23 = a5;
      v24 = v12;
      v13 = *(_BYTE **)(a4 + 8);
      v14 = 1;
      v15 = a4;
      v16 = *((_QWORD *)a2 + 7);
      do
      {
        if ( *(_BYTE **)(v15 + 16) == v13 )
        {
          sub_9CA200(v15, v13, (_QWORD *)(v16 + 32));
        }
        else
        {
          if ( v13 )
          {
            *(_QWORD *)v13 = *(_QWORD *)(v16 + 32);
            v13 = *(_BYTE **)(v15 + 8);
          }
          *(_QWORD *)(v15 + 8) = v13 + 8;
        }
        v14 &= sub_1039CC0(a1, *(_QWORD *)(v16 + 40), (_DWORD)a3, v15, v23, v24, (__int64)(a2 + 1));
        v13 = (_BYTE *)(*(_QWORD *)(v15 + 8) - 8LL);
        *(_QWORD *)(v15 + 8) = v13;
        v16 = sub_220EEE0(v16);
      }
      while ( a2 + 40 != (unsigned __int8 *)v16 );
      a4 = v15;
      a5 = v23;
      v9 = a2;
      if ( !(_BYTE)v14 )
      {
LABEL_11:
        v17 = 0;
        if ( a6 )
        {
          v28 = 0;
          v29 = 0;
          v30 = 0;
          sub_10394D0(a1, (__int64)v9, (__int64)&v28);
          v19 = sub_1039260(
                  a3,
                  *(__int64 **)a4,
                  (__int64)(*(_QWORD *)(a4 + 8) - *(_QWORD *)a4) >> 3,
                  1,
                  v28,
                  (v29 - v28) >> 4);
          v20 = *(_BYTE **)(a5 + 8);
          v27 = v19;
          if ( v20 == *(_BYTE **)(a5 + 16) )
          {
            sub_914280(a5, v20, &v27);
          }
          else
          {
            if ( v20 )
            {
              *(_QWORD *)v20 = v19;
              v20 = *(_BYTE **)(a5 + 8);
            }
            *(_QWORD *)(a5 + 8) = v20 + 8;
          }
          if ( v28 )
            j_j___libc_free_0(v28, v30 - v28);
          return a6;
        }
        return v17;
      }
    }
    return 1;
  }
  v17 = v10;
  if ( (*a2 & 2) == 0 && !*a7 && !(_BYTE)qword_4F8F368 )
    return 1;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  sub_10394D0(a1, (__int64)a2, (__int64)&v28);
  v21 = sub_1039260(
          a3,
          *(__int64 **)a4,
          (__int64)(*(_QWORD *)(a4 + 8) - *(_QWORD *)a4) >> 3,
          *a2,
          v28,
          (v29 - v28) >> 4);
  v22 = *(_BYTE **)(a5 + 8);
  v27 = v21;
  if ( v22 == *(_BYTE **)(a5 + 16) )
  {
    sub_914280(a5, v22, &v27);
  }
  else
  {
    if ( v22 )
    {
      *(_QWORD *)v22 = v21;
      v22 = *(_BYTE **)(a5 + 8);
    }
    *(_QWORD *)(a5 + 8) = v22 + 8;
  }
  if ( (*v9 & 2) == 0 )
    *a7 = 0;
  if ( v28 )
    j_j___libc_free_0(v28, v30 - v28);
  return v17;
}
