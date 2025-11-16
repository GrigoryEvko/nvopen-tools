// Function: sub_2EE9380
// Address: 0x2ee9380
//
__int64 __fastcall sub_2EE9380(
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
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // r14d
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r15
  unsigned int v18; // ebx
  __int64 *v19; // r12
  __int64 v20; // rax
  _QWORD *v21; // r11
  _WORD *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rsi
  unsigned __int16 *v25; // rax
  unsigned __int16 *i; // rcx
  _QWORD *v27; // r11
  _WORD *v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rsi
  unsigned __int16 *v31; // rax
  unsigned __int16 *j; // rcx
  __int64 v33; // rcx
  __int64 v34; // rdx
  unsigned int v35; // r12d
  int v36; // r14d
  __int64 *v37; // r13
  __int64 v38; // rsi
  unsigned int v39; // ecx
  __int64 result; // rax
  __int64 v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+20h] [rbp-70h]
  _QWORD *v44; // [rsp+28h] [rbp-68h]
  _QWORD *v45; // [rsp+30h] [rbp-60h]
  __int64 *v47; // [rsp+40h] [rbp-50h]
  __int64 v48; // [rsp+48h] [rbp-48h]
  unsigned int v50; // [rsp+5Ch] [rbp-34h]

  v8 = a1;
  v43 = sub_2EE8970(*a1, -1171354717 * (unsigned int)((a1[1] - *(_QWORD *)(*a1 + 8)) >> 3));
  v48 = v10;
  v42 = sub_2EE8AC0(*v8, -1171354717 * (unsigned int)((v8[1] - *(_QWORD *)(*v8 + 8)) >> 3));
  v47 = &a2[a3];
  if ( v48 )
  {
    v50 = 0;
    v45 = &a7[a8];
    v13 = 0;
    v44 = &a4[a5];
    v15 = *(_QWORD *)(*a1 + 440);
    v16 = 0;
    do
    {
      v17 = 4 * v16;
      v18 = *(_DWORD *)(v42 + 4 * v16) + *(_DWORD *)(v43 + 4 * v16);
      if ( a2 != v47 )
      {
        v19 = a2;
        while ( 1 )
        {
          v20 = *v19++;
          v18 += *(_DWORD *)(v17 + sub_2EE8550(v15, *(_DWORD *)(v20 + 24)));
          if ( v47 == v19 )
            break;
          v15 = *(_QWORD *)(*a1 + 440);
        }
        v15 = *(_QWORD *)(*a1 + 440);
      }
      if ( a4 != v44 )
      {
        v21 = a4;
        v12 = 0;
        do
        {
          v22 = (_WORD *)*v21;
          if ( (*(_WORD *)*v21 & 0x1FFF) != 0x1FFF )
          {
            v23 = (unsigned __int16)v22[1];
            v24 = *(_QWORD *)(*(_QWORD *)(v15 + 232) + 176LL);
            v25 = (unsigned __int16 *)(v24 + 6 * v23);
            for ( i = (unsigned __int16 *)(v24 + 6 * (v23 + (unsigned __int16)v22[2])); i != v25; v25 += 3 )
            {
              if ( *v25 == v13 )
                v12 = *(_DWORD *)(*(_QWORD *)(v15 + 248) + v17) * v25[1] + (unsigned int)v12;
            }
          }
          ++v21;
        }
        while ( v44 != v21 );
        v18 += v12;
      }
      v11 = (__int64)&a7[a8];
      if ( a7 != v45 )
      {
        v27 = a7;
        v12 = 0;
        do
        {
          v28 = (_WORD *)*v27;
          if ( (*(_WORD *)*v27 & 0x1FFF) != 0x1FFF )
          {
            v29 = (unsigned __int16)v28[1];
            v30 = *(_QWORD *)(*(_QWORD *)(v15 + 232) + 176LL);
            v31 = (unsigned __int16 *)(v30 + 6 * v29);
            for ( j = (unsigned __int16 *)(v30 + 6 * (v29 + (unsigned __int16)v28[2])); j != v31; v31 += 3 )
            {
              if ( *v31 == v13 )
                v12 = *(_DWORD *)(*(_QWORD *)(v15 + 248) + v17) * v31[1] + (unsigned int)v12;
            }
          }
          ++v27;
        }
        while ( v45 != v27 );
        v18 -= v12;
      }
      if ( v50 >= v18 )
        v18 = v50;
      v16 = (unsigned int)++v13;
      v50 = v18;
    }
    while ( v13 != v48 );
    v8 = a1;
  }
  else
  {
    v50 = 0;
    v15 = *(_QWORD *)(*a1 + 440);
  }
  v33 = *(unsigned int *)(v15 + 332);
  v34 = (v50 + (unsigned int)v33 - 1) % (unsigned int)v33;
  v35 = (v50 + (unsigned int)v33 - 1) / (unsigned int)v33;
  v36 = *(_DWORD *)(v8[1] + 24) + *(_DWORD *)(v8[1] + 28);
  if ( a2 != v47 )
  {
    v37 = a2;
    do
    {
      v38 = *v37++;
      v36 += *(_DWORD *)sub_2EE8230(v15, v38, v34, v33, v11, v12);
      v15 = *(_QWORD *)(*v8 + 440);
    }
    while ( v47 != v37 );
  }
  v39 = *(_DWORD *)(v15 + 40);
  result = (unsigned int)(v36 + a5 - a8);
  if ( v39 )
    result = (unsigned int)result / v39;
  if ( v35 >= (unsigned int)result )
    return v35;
  return result;
}
