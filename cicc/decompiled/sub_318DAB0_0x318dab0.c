// Function: sub_318DAB0
// Address: 0x318dab0
//
void __fastcall sub_318DAB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rdx
  _QWORD *v9; // rdi
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // r9
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // r8
  _QWORD *v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 *v21; // r8
  __int64 *v22; // r12
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // r13
  _BYTE *v26; // rdi
  _QWORD *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rsi
  _BYTE *v33; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v34; // [rsp+28h] [rbp-C8h]
  _BYTE v35[48]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD *v36; // [rsp+60h] [rbp-90h]
  _BYTE *v37; // [rsp+70h] [rbp-80h] BYREF
  __int64 v38; // [rsp+78h] [rbp-78h]
  _BYTE v39[48]; // [rsp+80h] [rbp-70h] BYREF
  _QWORD *v40; // [rsp+B0h] [rbp-40h]

  v7 = *(_QWORD *)(a1 + 8);
  v33 = v35;
  v34 = 0x600000000LL;
  v8 = *(unsigned int *)(v7 + 8);
  if ( (_DWORD)v8 )
    sub_318D930((__int64)&v33, v7, v8, a4, a5, a6);
  v9 = *(_QWORD **)(v7 + 64);
  v10 = *(_QWORD *)(a1 + 96);
  v36 = v9;
  v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v10 & 4) != 0 || !v11 )
    sub_B44240(v9, v11, (unsigned __int64 *)(v11 + 48), 0);
  else
    sub_B44220(v9, v11 + 24, 0);
  v15 = (unsigned __int64)v33;
  if ( &v33[8 * (unsigned int)v34] != v33 )
  {
    v13 = 0;
    v16 = (8 * (unsigned __int64)(unsigned int)v34 - 8) >> 3;
    while ( 1 )
    {
      v12 = *(_QWORD *)(v15 + 8 * v13);
      if ( (*((_BYTE *)v36 + 7) & 0x40) != 0 )
        v17 = (_QWORD *)*(v36 - 1);
      else
        v17 = &v36[-4 * (*((_DWORD *)v36 + 1) & 0x7FFFFFF)];
      v18 = &v17[4 * (unsigned int)v13];
      if ( *v18 )
      {
        v14 = (_QWORD *)v18[2];
        v19 = v18[1];
        *v14 = v19;
        if ( v19 )
        {
          v14 = (_QWORD *)v18[2];
          *(_QWORD *)(v19 + 16) = v14;
        }
      }
      *v18 = v12;
      if ( v12 )
      {
        v20 = *(_QWORD *)(v12 + 16);
        v14 = (_QWORD *)(v12 + 16);
        v18[1] = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = v18 + 1;
        v18[2] = (__int64)v14;
        *(_QWORD *)(v12 + 16) = v18;
      }
      if ( v16 == v13 )
        break;
      ++v13;
    }
  }
  v21 = *(__int64 **)(a1 + 8);
  v22 = &v21[9 * *(unsigned int *)(a1 + 16)];
  if ( v22 != v21 + 9 )
  {
    v23 = v6;
    v24 = (__int64)(v21 + 9);
    v25 = v23;
    do
    {
      v37 = v39;
      v38 = 0x600000000LL;
      if ( *(_DWORD *)(v24 + 8) )
        sub_318D930((__int64)&v37, v24, v12, v13, (__int64)v21, (__int64)v14);
      LOWORD(v25) = 0;
      v40 = *(_QWORD **)(v24 + 64);
      sub_B44220(v40, (__int64)(v36 + 3), v25);
      v26 = v37;
      if ( &v37[8 * (unsigned int)v38] != v37 )
      {
        v13 = 0;
        v14 = (_QWORD *)((8 * (unsigned __int64)(unsigned int)v38 - 8) >> 3);
        while ( 1 )
        {
          v12 = *(_QWORD *)&v26[8 * v13];
          if ( (*((_BYTE *)v40 + 7) & 0x40) != 0 )
            v27 = (_QWORD *)*(v40 - 1);
          else
            v27 = &v40[-4 * (*((_DWORD *)v40 + 1) & 0x7FFFFFF)];
          v28 = &v27[4 * (unsigned int)v13];
          if ( *v28 )
          {
            v21 = (__int64 *)v28[2];
            v29 = v28[1];
            *v21 = v29;
            if ( v29 )
            {
              v21 = (__int64 *)v28[2];
              *(_QWORD *)(v29 + 16) = v21;
            }
          }
          *v28 = v12;
          if ( v12 )
          {
            v30 = *(_QWORD *)(v12 + 16);
            v21 = (__int64 *)(v12 + 16);
            v28[1] = v30;
            if ( v30 )
              *(_QWORD *)(v30 + 16) = v28 + 1;
            v28[2] = (__int64)v21;
            *(_QWORD *)(v12 + 16) = v28;
          }
          if ( (_QWORD *)v13 == v14 )
            break;
          ++v13;
        }
        v26 = v37;
      }
      v36 = v40;
      if ( v26 != v39 )
        _libc_free((unsigned __int64)v26);
      v24 += 72;
    }
    while ( (__int64 *)v24 != v22 );
  }
  sub_3189570(*(_QWORD *)(a2 + 72), a1 + 104);
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
}
