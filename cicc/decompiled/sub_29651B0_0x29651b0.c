// Function: sub_29651B0
// Address: 0x29651b0
//
_QWORD *__fastcall sub_29651B0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  _BYTE *v8; // rsi
  __int64 v9; // r9
  __int64 v10; // r14
  __int64 *v11; // r14
  __int64 v12; // rdx
  __int64 *v13; // r15
  unsigned __int64 v14; // rcx
  __int64 v15; // r12
  char *v16; // rdx
  __int64 *v17; // rbx
  char *v18; // rax
  __int64 v19; // r15
  __int64 v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // r8
  _QWORD *v23; // r14
  _BYTE *v24; // rsi
  __int64 v25; // r9
  __int64 v26; // r15
  __int64 v27; // r12
  _QWORD *v28; // rbx
  __int64 v29; // r14
  char *v30; // rdx
  __int64 v32; // [rsp+8h] [rbp-188h]
  _QWORD *v33; // [rsp+10h] [rbp-180h]
  __int64 *v34; // [rsp+28h] [rbp-168h]
  _QWORD *v35; // [rsp+38h] [rbp-158h] BYREF
  __int64 v36[2]; // [rsp+40h] [rbp-150h] BYREF
  char *v37; // [rsp+50h] [rbp-140h] BYREF
  __int64 v38; // [rsp+58h] [rbp-138h]
  _BYTE v39[304]; // [rsp+60h] [rbp-130h] BYREF

  v32 = (__int64)(a4 + 7);
  v6 = a4[7];
  v36[0] = a3;
  v36[1] = (__int64)a4;
  a4[17] += 160LL;
  v7 = ((v6 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 160;
  if ( a4[8] >= v7 && v6 )
  {
    a4[7] = v7;
    v33 = (_QWORD *)((v6 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  }
  else
  {
    v33 = (_QWORD *)sub_9D1E70(v32, 160, 160, 3);
  }
  memset(v33, 0, 0xA0u);
  v33[9] = 8;
  v33[8] = v33 + 11;
  *((_BYTE *)v33 + 84) = 1;
  if ( a2 )
  {
    v37 = (char *)v33;
    *v33 = a2;
    v8 = *(_BYTE **)(a2 + 16);
    if ( v8 == *(_BYTE **)(a2 + 24) )
    {
      sub_D4C7F0(a2 + 8, v8, &v37);
    }
    else
    {
      if ( v8 )
      {
        *(_QWORD *)v8 = v37;
        v8 = *(_BYTE **)(a2 + 16);
      }
      *(_QWORD *)(a2 + 16) = v8 + 8;
    }
  }
  else
  {
    v37 = (char *)v33;
    sub_D4C980((__int64)(a4 + 4), &v37);
  }
  sub_29638A0(v36, a1, (__int64)v33);
  v10 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 8) != v10 )
  {
    v11 = (__int64 *)(v10 - 8);
    v12 = 0;
    v13 = *(__int64 **)(a1 + 8);
    v14 = 16;
    v37 = v39;
    v38 = 0x1000000000LL;
    while ( 1 )
    {
      v15 = *v11;
      if ( v12 + 1 > v14 )
      {
        sub_C8D5F0((__int64)&v37, v39, v12 + 1, 0x10u, v12 + 1, v9);
        v12 = (unsigned int)v38;
      }
      v16 = &v37[16 * v12];
      *(_QWORD *)v16 = v33;
      *((_QWORD *)v16 + 1) = v15;
      v12 = (unsigned int)(v38 + 1);
      LODWORD(v38) = v38 + 1;
      if ( v13 == v11 )
        break;
      v14 = HIDWORD(v38);
      --v11;
    }
    v17 = v36;
    do
    {
      v18 = &v37[16 * (unsigned int)v12 - 16];
      v19 = *(_QWORD *)v18;
      v20 = *((_QWORD *)v18 + 1);
      LODWORD(v38) = v12 - 1;
      a4[17] += 160LL;
      v21 = a4[7];
      v22 = (v21 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a4[8] >= v22 + 160 && v21 )
      {
        a4[7] = v22 + 160;
        v23 = (_QWORD *)((v21 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        v23 = (_QWORD *)sub_9D1E70(v32, 160, 160, 3);
      }
      memset(v23, 0, 0xA0u);
      *((_BYTE *)v23 + 84) = 1;
      v23[8] = v23 + 11;
      v23[9] = 8;
      v35 = v23;
      *v23 = v19;
      v24 = *(_BYTE **)(v19 + 16);
      if ( v24 == *(_BYTE **)(v19 + 24) )
      {
        sub_D4C7F0(v19 + 8, v24, &v35);
      }
      else
      {
        if ( v24 )
        {
          *(_QWORD *)v24 = v35;
          v24 = *(_BYTE **)(v19 + 16);
        }
        *(_QWORD *)(v19 + 16) = v24 + 8;
      }
      sub_29638A0(v17, v20, (__int64)v23);
      v26 = *(_QWORD *)(v20 + 8);
      v27 = *(_QWORD *)(v20 + 16);
      if ( v26 == v27 )
      {
        LODWORD(v12) = v38;
      }
      else
      {
        v12 = (unsigned int)v38;
        v34 = v17;
        v28 = v23;
        do
        {
          v29 = *(_QWORD *)(v27 - 8);
          if ( v12 + 1 > (unsigned __int64)HIDWORD(v38) )
          {
            sub_C8D5F0((__int64)&v37, v39, v12 + 1, 0x10u, v12 + 1, v25);
            v12 = (unsigned int)v38;
          }
          v30 = &v37[16 * v12];
          v27 -= 8;
          *(_QWORD *)v30 = v28;
          *((_QWORD *)v30 + 1) = v29;
          v12 = (unsigned int)(v38 + 1);
          LODWORD(v38) = v38 + 1;
        }
        while ( v26 != v27 );
        v17 = v34;
      }
    }
    while ( (_DWORD)v12 );
    if ( v37 != v39 )
      _libc_free((unsigned __int64)v37);
  }
  return v33;
}
