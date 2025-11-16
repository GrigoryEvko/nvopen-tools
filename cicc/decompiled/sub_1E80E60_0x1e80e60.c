// Function: sub_1E80E60
// Address: 0x1e80e60
//
__int64 __fastcall sub_1E80E60(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8)
{
  __int64 *v8; // r15
  __int64 v10; // rdx
  int v11; // r8d
  int v12; // r14d
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r15
  unsigned int v17; // ebx
  __int64 *v18; // r12
  __int64 v19; // rax
  _QWORD *v20; // r11
  int v21; // r9d
  _WORD *v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rcx
  unsigned __int16 *v25; // rax
  unsigned __int16 *i; // rcx
  _QWORD *v27; // r11
  int v28; // r9d
  _WORD *v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rcx
  unsigned __int16 *v32; // rax
  unsigned __int16 *j; // rcx
  __int64 v34; // rcx
  __int64 v35; // rdx
  unsigned int v36; // r12d
  int v37; // r14d
  __int64 *v38; // r13
  __int64 v39; // rsi
  unsigned int v40; // ecx
  __int64 result; // rax
  __int64 v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+20h] [rbp-70h]
  _QWORD *v45; // [rsp+28h] [rbp-68h]
  _QWORD *v46; // [rsp+30h] [rbp-60h]
  __int64 *v48; // [rsp+40h] [rbp-50h]
  __int64 v49; // [rsp+48h] [rbp-48h]
  unsigned int v51; // [rsp+5Ch] [rbp-34h]

  v8 = a1;
  v44 = sub_1E80530(*a1, -1171354717 * (unsigned int)((a1[1] - *(_QWORD *)(*a1 + 8)) >> 3));
  v49 = v10;
  v43 = sub_1E80680(*v8, -1171354717 * (unsigned int)((v8[1] - *(_QWORD *)(*v8 + 8)) >> 3));
  v48 = &a2[a3];
  if ( v49 )
  {
    v51 = 0;
    v46 = &a7[a8];
    v12 = 0;
    v45 = &a4[a5];
    v14 = *(_QWORD *)(*a1 + 440);
    v15 = 0;
    do
    {
      v16 = 4 * v15;
      v17 = *(_DWORD *)(v43 + 4 * v15) + *(_DWORD *)(v44 + 4 * v15);
      if ( a2 != v48 )
      {
        v18 = a2;
        while ( 1 )
        {
          v19 = *v18++;
          v17 += *(_DWORD *)(v16 + sub_1E80160(v14, *(_DWORD *)(v19 + 48)));
          if ( v48 == v18 )
            break;
          v14 = *(_QWORD *)(*a1 + 440);
        }
        v14 = *(_QWORD *)(*a1 + 440);
      }
      if ( a4 != v45 )
      {
        v20 = a4;
        v21 = 0;
        do
        {
          v22 = (_WORD *)*v20;
          if ( (*(_WORD *)*v20 & 0x3FFF) != 0x3FFF )
          {
            v23 = (unsigned __int16)v22[1];
            v24 = *(_QWORD *)(*(_QWORD *)(v14 + 448) + 136LL);
            v25 = (unsigned __int16 *)(v24 + 4 * v23);
            for ( i = (unsigned __int16 *)(v24 + 4 * (v23 + (unsigned __int16)v22[2])); i != v25; v25 += 2 )
            {
              if ( *v25 == v12 )
                v21 += *(_DWORD *)(*(_QWORD *)(v14 + 464) + v16) * v25[1];
            }
          }
          ++v20;
        }
        while ( v45 != v20 );
        v17 += v21;
      }
      v11 = (_DWORD)a7 + 8 * a8;
      if ( a7 != v46 )
      {
        v27 = a7;
        v28 = 0;
        do
        {
          v29 = (_WORD *)*v27;
          if ( (*(_WORD *)*v27 & 0x3FFF) != 0x3FFF )
          {
            v30 = (unsigned __int16)v29[1];
            v31 = *(_QWORD *)(*(_QWORD *)(v14 + 448) + 136LL);
            v32 = (unsigned __int16 *)(v31 + 4 * v30);
            for ( j = (unsigned __int16 *)(v31 + 4 * (v30 + (unsigned __int16)v29[2])); j != v32; v32 += 2 )
            {
              if ( *v32 == v12 )
                v28 += *(_DWORD *)(*(_QWORD *)(v14 + 464) + v16) * v32[1];
            }
          }
          ++v27;
        }
        while ( v46 != v27 );
        v17 -= v28;
      }
      if ( v51 >= v17 )
        v17 = v51;
      v15 = (unsigned int)++v12;
      v51 = v17;
    }
    while ( v12 != v49 );
    v8 = a1;
  }
  else
  {
    v51 = 0;
    v14 = *(_QWORD *)(*a1 + 440);
  }
  v34 = *(unsigned int *)(v14 + 548);
  v35 = (v51 + (unsigned int)v34 - 1) % (unsigned int)v34;
  v36 = (v51 + (unsigned int)v34 - 1) / (unsigned int)v34;
  v37 = *(_DWORD *)(v8[1] + 24) + *(_DWORD *)(v8[1] + 28);
  if ( a2 != v48 )
  {
    v38 = a2;
    do
    {
      v39 = *v38++;
      v37 += *(_DWORD *)sub_1E7FE90(v14, v39, v35, v34, v11);
      v14 = *(_QWORD *)(*v8 + 440);
    }
    while ( v48 != v38 );
  }
  v40 = *(_DWORD *)(v14 + 272);
  result = (unsigned int)(v37 + a5 - a8);
  if ( v40 )
    result = (unsigned int)result / v40;
  if ( v36 >= (unsigned int)result )
    return v36;
  return result;
}
