// Function: sub_385C5C0
// Address: 0x385c5c0
//
_QWORD *__fastcall sub_385C5C0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __int64 a9)
{
  __int64 v12; // rbx
  __int64 v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rcx
  _QWORD *v16; // rax
  _QWORD *v17; // rcx
  __int64 v18; // rax
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rsi
  _QWORD *v28; // rdx
  _QWORD *v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  __int64 v32; // [rsp+20h] [rbp-60h]
  int v33; // [rsp+28h] [rbp-58h]
  __int64 v34; // [rsp+28h] [rbp-58h]
  __int64 v35[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v36[8]; // [rsp+40h] [rbp-40h] BYREF

  v12 = *(_QWORD *)(*(_QWORD *)(a9 + 8) + ((unsigned __int64)**(unsigned int **)(a2 + 24) << 6) + 16);
  v32 = sub_146F1B0(a6, v12);
  v13 = *(_QWORD *)v12;
  if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16 )
    v13 = **(_QWORD **)(v13 + 16);
  v33 = *(_DWORD *)(v13 + 8) >> 8;
  v14 = (_QWORD *)sub_16498A0(a4);
  v34 = sub_16471D0(v14, v33);
  if ( !sub_146CEE0(a6, v32, a3) )
  {
    v23 = sub_38767A0(a5, *(_QWORD *)(a2 + 16), v34, a4);
    v24 = sub_38767A0(a5, *(_QWORD *)(a2 + 8), v34, a4);
    *a1 = 6;
    a1[1] = 0;
    v25 = v24;
    if ( v23 )
    {
      a1[2] = v23;
      if ( v23 != -16 && v23 != -8 )
        sub_164C220((__int64)a1);
    }
    else
    {
      a1[2] = 0;
    }
    a1[3] = 6;
    a1[4] = 0;
    if ( v25 )
    {
      a1[5] = v25;
      if ( v25 != -16 && v25 != -8 )
        goto LABEL_20;
      return a1;
    }
LABEL_43:
    a1[5] = 0;
    return a1;
  }
  if ( *(_BYTE *)(v12 + 16) > 0x17u )
  {
    v15 = *(_QWORD **)(a3 + 72);
    v16 = *(_QWORD **)(a3 + 64);
    if ( v15 == v16 )
    {
      v17 = &v16[*(unsigned int *)(a3 + 84)];
      if ( v16 == v17 )
      {
        v28 = *(_QWORD **)(a3 + 64);
      }
      else
      {
        do
        {
          if ( *(_QWORD *)(v12 + 40) == *v16 )
            break;
          ++v16;
        }
        while ( v17 != v16 );
        v28 = v17;
      }
    }
    else
    {
      v30 = *(_QWORD *)(v12 + 40);
      v29 = &v15[*(unsigned int *)(a3 + 80)];
      v16 = sub_16CC9F0(a3 + 56, v30);
      v17 = v29;
      if ( v30 == *v16 )
      {
        v26 = *(_QWORD *)(a3 + 72);
        if ( v26 == *(_QWORD *)(a3 + 64) )
          v27 = *(unsigned int *)(a3 + 84);
        else
          v27 = *(unsigned int *)(a3 + 80);
        v28 = (_QWORD *)(v26 + 8 * v27);
      }
      else
      {
        v18 = *(_QWORD *)(a3 + 72);
        if ( v18 != *(_QWORD *)(a3 + 64) )
        {
          v16 = (_QWORD *)(v18 + 8LL * *(unsigned int *)(a3 + 80));
          goto LABEL_9;
        }
        v16 = (_QWORD *)(v18 + 8LL * *(unsigned int *)(a3 + 84));
        v28 = v16;
      }
    }
    while ( v28 != v16 && *v16 >= 0xFFFFFFFFFFFFFFFELL )
      ++v16;
LABEL_9:
    if ( v16 != v17 )
      v12 = sub_38767A0(a5, v32, v34, a4);
  }
  v36[1] = sub_145CF80(a6, v34, 1, 0);
  v36[0] = v32;
  v35[0] = (__int64)v36;
  v35[1] = 0x200000002LL;
  v19 = sub_147DD40(a6, v35, 0, 0, a7, a8);
  if ( (_QWORD *)v35[0] != v36 )
    _libc_free(v35[0]);
  v20 = sub_38767A0(a5, v19, v34, a4);
  *a1 = 6;
  a1[1] = 0;
  v21 = v20;
  if ( v12 )
  {
    a1[2] = v12;
    if ( v12 != -16 && v12 != -8 )
      sub_164C220((__int64)a1);
  }
  else
  {
    a1[2] = 0;
  }
  a1[3] = 6;
  a1[4] = 0;
  if ( !v21 )
    goto LABEL_43;
  a1[5] = v21;
  if ( v21 != -8 && v21 != -16 )
LABEL_20:
    sub_164C220((__int64)(a1 + 3));
  return a1;
}
