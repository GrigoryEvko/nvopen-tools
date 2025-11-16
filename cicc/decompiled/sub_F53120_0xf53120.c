// Function: sub_F53120
// Address: 0xf53120
//
void __fastcall sub_F53120(_BYTE *a1, unsigned __int8 *a2, __int64 a3, int a4)
{
  __int64 v4; // rsi
  __int64 *v5; // rbx
  __int64 *v6; // r14
  __int64 v7; // r12
  _QWORD *v8; // r8
  _QWORD *v9; // rdx
  __int64 *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 *v15; // rdi
  __int64 *v16; // rbx
  _QWORD *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rsi
  unsigned __int8 *v20; // rsi
  __int64 v21; // r12
  __int64 v22; // r13
  _QWORD *v23; // r14
  __int64 *v26; // [rsp+28h] [rbp-88h]
  __int64 v27; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int8 *v28; // [rsp+38h] [rbp-78h] BYREF
  __int64 *v29; // [rsp+40h] [rbp-70h] BYREF
  __int64 v30; // [rsp+48h] [rbp-68h]
  _BYTE v31[16]; // [rsp+50h] [rbp-60h] BYREF
  __int64 *v32; // [rsp+60h] [rbp-50h] BYREF
  __int64 v33; // [rsp+68h] [rbp-48h]
  _BYTE v34[64]; // [rsp+70h] [rbp-40h] BYREF

  v4 = (__int64)a1;
  v29 = (__int64 *)v31;
  v30 = 0x100000000LL;
  v33 = 0x100000000LL;
  v32 = (__int64 *)v34;
  sub_AE7A40((__int64)&v29, a1, (__int64)&v32);
  v5 = v29;
  v6 = &v29[(unsigned int)v30];
  if ( v29 != v6 )
  {
    do
    {
      v7 = *v5;
      v8 = *(_QWORD **)(*(_QWORD *)(*v5 + 32 * (2LL - (*(_DWORD *)(*v5 + 4) & 0x7FFFFFF))) + 24LL);
      if ( v8 )
      {
        v9 = (_QWORD *)v8[2];
        if ( (unsigned int)((__int64)(v8[3] - (_QWORD)v9) >> 3) )
        {
          if ( *v9 == 6 )
          {
            if ( a4 )
              v8 = (_QWORD *)sub_B0DAC0(v8, 0, a4);
            v10 = (__int64 *)(v8[1] & 0xFFFFFFFFFFFFFFF8LL);
            if ( (v8[1] & 4) != 0 )
              v10 = (__int64 *)*v10;
            v11 = sub_B9F6F0(v10, v8);
            v12 = v7 + 32 * (2LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
            if ( *(_QWORD *)v12 )
            {
              v13 = *(_QWORD *)(v12 + 8);
              **(_QWORD **)(v12 + 16) = v13;
              if ( v13 )
                *(_QWORD *)(v13 + 16) = *(_QWORD *)(v12 + 16);
            }
            *(_QWORD *)v12 = v11;
            if ( v11 )
            {
              v14 = *(_QWORD *)(v11 + 16);
              *(_QWORD *)(v12 + 8) = v14;
              if ( v14 )
                *(_QWORD *)(v14 + 16) = v12 + 8;
              *(_QWORD *)(v12 + 16) = v11 + 16;
              *(_QWORD *)(v11 + 16) = v12;
            }
            v4 = 0;
            sub_B58F60(v7, 0, a2);
          }
        }
      }
      ++v5;
    }
    while ( v6 != v5 );
  }
  v15 = v32;
  v16 = v32;
  v26 = &v32[(unsigned int)v33];
  if ( v26 != v32 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v21 = *v16;
        v22 = *v16 + 80;
        v23 = (_QWORD *)sub_B11F60(v22);
        sub_B12000(v21 + 72);
        v4 = *(_QWORD *)(v21 + 24);
        v27 = v4;
        if ( !v4 )
          break;
        sub_B96E90((__int64)&v27, v4, 1);
        if ( v23 )
          goto LABEL_21;
LABEL_30:
        v4 = v27;
        if ( v27 )
          sub_B91220((__int64)&v27, v27);
        if ( v26 == ++v16 )
        {
LABEL_36:
          v15 = v32;
          goto LABEL_37;
        }
      }
      if ( v23 )
      {
LABEL_21:
        v17 = (_QWORD *)v23[2];
        if ( (unsigned int)((__int64)(v23[3] - (_QWORD)v17) >> 3) && *v17 == 6 )
        {
          if ( a4 )
            v23 = (_QWORD *)sub_B0DAC0(v23, 0, a4);
          sub_B11F20(&v28, (__int64)v23);
          v19 = *(_QWORD *)(v21 + 80);
          if ( v19 )
            sub_B91220(v22, v19);
          v20 = v28;
          *(_QWORD *)(v21 + 80) = v28;
          if ( v20 )
            sub_B976B0((__int64)&v28, v20, v22);
          sub_B12AA0(v21, 0, (char *)a2, v18);
        }
        goto LABEL_30;
      }
      if ( v26 == ++v16 )
        goto LABEL_36;
    }
  }
LABEL_37:
  if ( v15 != (__int64 *)v34 )
    _libc_free(v15, v4);
  if ( v29 != (__int64 *)v31 )
    _libc_free(v29, v4);
}
