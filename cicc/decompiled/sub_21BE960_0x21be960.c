// Function: sub_21BE960
// Address: 0x21be960
//
__int64 __fastcall sub_21BE960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r11d
  unsigned int *v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // r8
  __int64 **v12; // r11
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 *v20; // r12
  bool v22; // zf
  __int16 v23; // bx
  __int64 v24; // rsi
  _QWORD *v25; // r11
  __int64 v26; // rbx
  __int64 *v27; // r13
  __int64 *v28; // r12
  __int64 v29; // rsi
  __int64 *v30; // r14
  __int64 v31; // rsi
  __int64 *v32; // rax
  __int64 v33; // rax
  __int128 v34; // [rsp-10h] [rbp-E0h]
  __int64 v35; // [rsp+0h] [rbp-D0h]
  __int64 v36; // [rsp+0h] [rbp-D0h]
  __int64 **v37; // [rsp+0h] [rbp-D0h]
  _QWORD *v38; // [rsp+8h] [rbp-C8h]
  __int64 **v39; // [rsp+8h] [rbp-C8h]
  __int64 v40; // [rsp+8h] [rbp-C8h]
  unsigned __int8 v41; // [rsp+10h] [rbp-C0h]
  unsigned __int8 v42; // [rsp+18h] [rbp-B8h]
  __int64 v43; // [rsp+20h] [rbp-B0h] BYREF
  int v44; // [rsp+28h] [rbp-A8h]
  __int64 v45; // [rsp+30h] [rbp-A0h] BYREF
  int v46; // [rsp+38h] [rbp-98h]
  __int64 *v47; // [rsp+40h] [rbp-90h] BYREF
  __int64 v48; // [rsp+48h] [rbp-88h]
  _BYTE v49[32]; // [rsp+50h] [rbp-80h] BYREF
  __int64 *v50; // [rsp+70h] [rbp-60h] BYREF
  __int64 v51; // [rsp+78h] [rbp-58h]
  _BYTE v52[80]; // [rsp+80h] [rbp-50h] BYREF

  v6 = 0;
  v7 = *(unsigned int **)(a2 + 32);
  v8 = *(_QWORD *)v7;
  v9 = v7[2];
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16 * v9) == 86 )
  {
    v10 = *(_QWORD *)(v8 + 48);
    v47 = (__int64 *)v49;
    v48 = 0x400000000LL;
    v50 = (__int64 *)v52;
    v51 = 0x400000000LL;
    if ( v10 )
    {
      v11 = a2;
      v12 = &v50;
      do
      {
        v13 = *(_QWORD *)(v10 + 16);
        if ( *(_WORD *)(v13 + 24) == 106 )
        {
          v14 = *(_QWORD *)(v13 + 32);
          if ( *(_QWORD *)v14 == v8 && *(_DWORD *)(v14 + 8) == (_DWORD)v9 )
          {
            v15 = *(_QWORD *)(v14 + 40);
            v16 = *(unsigned __int16 *)(v15 + 24);
            if ( v16 == 10 || v16 == 32 )
            {
              v17 = *(_QWORD *)(v15 + 88);
              v18 = *(_QWORD **)(v17 + 24);
              if ( *(_DWORD *)(v17 + 32) > 0x40u )
                v18 = (_QWORD *)*v18;
              if ( v18 )
              {
                v19 = (unsigned int)v51;
                if ( (unsigned int)v51 >= HIDWORD(v51) )
                {
                  v36 = v11;
                  v39 = v12;
                  sub_16CD150((__int64)v12, v52, 0, 8, v11, a6);
                  v19 = (unsigned int)v51;
                  v11 = v36;
                  v12 = v39;
                }
                v50[v19] = v13;
                LODWORD(v51) = v51 + 1;
              }
              else
              {
                v33 = (unsigned int)v48;
                if ( (unsigned int)v48 >= HIDWORD(v48) )
                {
                  v37 = v12;
                  v40 = v11;
                  sub_16CD150((__int64)&v47, v49, 0, 8, v11, a6);
                  v33 = (unsigned int)v48;
                  v12 = v37;
                  v11 = v40;
                }
                v47[v33] = v13;
                LODWORD(v48) = v48 + 1;
              }
            }
          }
        }
        v10 = *(_QWORD *)(v10 + 32);
      }
      while ( v10 );
      if ( (_DWORD)v48 && (_DWORD)v51 )
      {
        v22 = *(_WORD *)(v8 + 24) == 158;
        v43 = v8;
        v23 = 4101;
        v44 = v9;
        if ( v22 )
        {
          v32 = *(__int64 **)(v8 + 32);
          v23 = 4102;
          v43 = *v32;
          v44 = *((_DWORD *)v32 + 2);
        }
        v24 = *(_QWORD *)(v11 + 72);
        v25 = *(_QWORD **)(a1 + 272);
        v45 = v24;
        if ( v24 )
        {
          v35 = v11;
          v38 = v25;
          sub_1623A60((__int64)&v45, v24, 2);
          v11 = v35;
          v25 = v38;
        }
        *((_QWORD *)&v34 + 1) = 1;
        *(_QWORD *)&v34 = &v43;
        v46 = *(_DWORD *)(v11 + 64);
        v26 = sub_1D25BD0(v25, v23, (__int64)&v45, 8, 0, a6, 8u, v34);
        if ( v45 )
          sub_161E7C0((__int64)&v45, v45);
        v27 = v47;
        v28 = &v47[(unsigned int)v48];
        if ( v47 != v28 )
        {
          do
          {
            v29 = *v27++;
            sub_1D44C70(*(_QWORD *)(a1 + 272), v29, 0, v26, 0);
            sub_1D49010(v26);
          }
          while ( v28 != v27 );
        }
        v20 = &v50[(unsigned int)v51];
        v30 = v50;
        if ( v50 == v20 )
        {
          v6 = 1;
        }
        else
        {
          do
          {
            v31 = *v30++;
            sub_1D44C70(*(_QWORD *)(a1 + 272), v31, 0, v26, 1u);
            sub_1D49010(v26);
          }
          while ( v20 != v30 );
          v20 = v50;
          v6 = 1;
        }
      }
      else
      {
        v20 = v50;
        v6 = 0;
      }
      if ( v20 != (__int64 *)v52 )
      {
        v41 = v6;
        _libc_free((unsigned __int64)v20);
        v6 = v41;
      }
    }
    if ( v47 != (__int64 *)v49 )
    {
      v42 = v6;
      _libc_free((unsigned __int64)v47);
      return v42;
    }
  }
  return v6;
}
