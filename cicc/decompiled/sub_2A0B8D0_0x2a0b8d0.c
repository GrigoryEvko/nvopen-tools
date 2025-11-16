// Function: sub_2A0B8D0
// Address: 0x2a0b8d0
//
__int64 __fastcall sub_2A0B8D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r13
  unsigned __int8 *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  unsigned __int8 *v16; // r8
  unsigned __int8 *v17; // rax
  size_t v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // r14
  __int64 v22; // rdx
  unsigned __int64 v23; // r15
  _BYTE *v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r13
  unsigned __int8 *v28; // rdx
  unsigned __int64 v29; // rbx
  unsigned __int8 *v30; // rdx
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v35; // rdx
  int v36; // edx
  int v37; // r9d
  __int64 v40; // [rsp+20h] [rbp-E0h]
  __int64 v41; // [rsp+28h] [rbp-D8h]
  __int64 v42; // [rsp+30h] [rbp-D0h]
  _QWORD *v43; // [rsp+30h] [rbp-D0h]
  __int64 v45; // [rsp+40h] [rbp-C0h]
  _BYTE *v47; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+58h] [rbp-A8h]
  _BYTE v49[16]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int64 v50; // [rsp+70h] [rbp-90h] BYREF
  __int64 v51; // [rsp+78h] [rbp-88h]
  _QWORD v52[2]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v53[14]; // [rsp+90h] [rbp-70h] BYREF

  v7 = *(_QWORD *)(a1 + 56);
  v40 = a1 + 48;
  while ( 1 )
  {
    if ( !v7 )
      BUG();
    if ( *(_BYTE *)(v7 - 24) != 84 )
      break;
    if ( (*(_DWORD *)(v7 - 20) & 0x7FFFFFF) != 0 )
    {
      v8 = 0;
      while ( 1 )
      {
        v9 = v8;
        if ( a2 == *(_QWORD *)(*(_QWORD *)(v7 - 32) + 32LL * *(unsigned int *)(v7 + 48) + 8 * v8) )
          break;
        if ( (*(_DWORD *)(v7 - 20) & 0x7FFFFFF) == (_DWORD)++v8 )
          goto LABEL_68;
      }
    }
    else
    {
LABEL_68:
      v9 = -1;
    }
    sub_B48BF0(v7 - 24, v9, 1);
    v7 = *(_QWORD *)(v7 + 8);
  }
  sub_11D2BF0((__int64)v53, a5);
  v10 = *(_QWORD *)(a1 + 56);
  if ( v40 != v10 )
  {
    while ( 1 )
    {
      if ( !v10 )
        BUG();
      if ( !*(_QWORD *)(v10 - 8) )
        goto LABEL_11;
      v11 = (unsigned __int8 *)(v10 - 24);
      v12 = *(unsigned int *)(a3 + 24);
      if ( (_DWORD)v12 )
      {
        v13 = *(_QWORD *)(a3 + 8);
        v14 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v15 = (_QWORD *)(v13 + ((unsigned __int64)v14 << 6));
        v16 = (unsigned __int8 *)v15[3];
        if ( v11 == v16 )
        {
LABEL_16:
          if ( v15 != (_QWORD *)(v13 + (v12 << 6)) )
          {
            v50 = 6;
            v51 = 0;
            v52[0] = v15[7];
            v45 = v52[0];
            if ( v45 != 0 && v45 != -4096 && v52[0] != -8192 )
            {
              sub_BD6050(&v50, v15[5] & 0xFFFFFFFFFFFFFFF8LL);
              v45 = v52[0];
              if ( v52[0] != -8192 && v52[0] != 0 && v52[0] != -4096 )
                sub_BD60C0(&v50);
            }
            goto LABEL_22;
          }
        }
        else
        {
          v36 = 1;
          while ( v16 != (unsigned __int8 *)-4096LL )
          {
            v37 = v36 + 1;
            v14 = (v12 - 1) & (v36 + v14);
            v15 = (_QWORD *)(v13 + ((unsigned __int64)v14 << 6));
            v16 = (unsigned __int8 *)v15[3];
            if ( v11 == v16 )
              goto LABEL_16;
            v36 = v37;
          }
        }
      }
      v45 = 0;
LABEL_22:
      v17 = (unsigned __int8 *)sub_BD5D20(v10 - 24);
      sub_11D2C80(v53, *(_QWORD *)(v10 - 16), v17, v18);
      if ( a4 )
        sub_DAC8D0(a4, (_BYTE *)(v10 - 24));
      sub_11D33F0(v53, a1, v10 - 24);
      sub_11D33F0(v53, a2, v45);
      v19 = *(_QWORD *)(v10 - 8);
      if ( v19 )
      {
        while ( 1 )
        {
          v20 = *(_QWORD *)(v19 + 24);
          v21 = *(_QWORD *)(v19 + 8);
          if ( *(_BYTE *)v20 == 84 )
            goto LABEL_30;
          v22 = *(_QWORD *)(v20 + 40);
          if ( a1 == v22 )
            goto LABEL_31;
          if ( a2 != v22 )
          {
LABEL_30:
            sub_11D9630(v53, v19);
            goto LABEL_31;
          }
          if ( *(_QWORD *)v19 )
          {
            **(_QWORD **)(v19 + 16) = v21;
            if ( v21 )
            {
              *(_QWORD *)(v21 + 16) = *(_QWORD *)(v19 + 16);
              *(_QWORD *)v19 = v45;
              if ( !v45 )
                goto LABEL_26;
            }
            else
            {
              *(_QWORD *)v19 = v45;
              if ( !v45 )
              {
LABEL_32:
                v11 = (unsigned __int8 *)(v10 - 24);
                break;
              }
            }
          }
          else
          {
            *(_QWORD *)v19 = v45;
            if ( !v45 )
              goto LABEL_31;
          }
          v35 = *(_QWORD *)(v45 + 16);
          *(_QWORD *)(v19 + 8) = v35;
          if ( v35 )
            *(_QWORD *)(v35 + 16) = v19 + 8;
          *(_QWORD *)(v19 + 16) = v45 + 16;
          *(_QWORD *)(v45 + 16) = v19;
LABEL_31:
          if ( !v21 )
            goto LABEL_32;
LABEL_26:
          v19 = v21;
        }
      }
      v47 = v49;
      v48 = 0x100000000LL;
      v51 = 0x100000000LL;
      v50 = (unsigned __int64)v52;
      sub_AE7A40((__int64)&v47, v11, (__int64)&v50);
      v23 = (unsigned __int64)v47;
      v24 = &v47[8 * (unsigned int)v48];
      if ( v47 != v24 )
      {
        v42 = v10;
        do
        {
          v26 = *(_QWORD *)v23;
          v27 = *(_QWORD *)(*(_QWORD *)v23 + 40LL);
          if ( a1 != v27 )
          {
            v28 = (unsigned __int8 *)v45;
            if ( a2 != v27 )
            {
              if ( (unsigned __int8)sub_11D3030(v53, *(_QWORD *)(*(_QWORD *)v23 + 40LL)) )
                v25 = sub_11D7E40(v53, v27);
              else
                v25 = sub_ACADE0(*(__int64 ***)(v42 - 16));
              v26 = *(_QWORD *)v23;
              v28 = (unsigned __int8 *)v25;
            }
            sub_B59720(v26, (__int64)v11, v28);
          }
          v23 += 8LL;
        }
        while ( v24 != (_BYTE *)v23 );
        v10 = v42;
      }
      v29 = v50;
      v43 = (_QWORD *)(v50 + 8LL * (unsigned int)v51);
      if ( (_QWORD *)v50 != v43 )
      {
        v41 = v10;
        do
        {
          v31 = *(_QWORD *)v29;
          v32 = sub_B14180(*(_QWORD *)(*(_QWORD *)v29 + 16LL));
          v33 = v32;
          if ( a1 != v32 )
          {
            v30 = (unsigned __int8 *)v45;
            if ( a2 != v32 )
            {
              if ( (unsigned __int8)sub_11D3030(v53, v32) )
                v30 = (unsigned __int8 *)sub_11D7E40(v53, v33);
              else
                v30 = (unsigned __int8 *)sub_ACADE0(*(__int64 ***)(v41 - 16));
            }
            sub_B13360(v31, v11, v30, 0);
          }
          v29 += 8LL;
        }
        while ( v43 != (_QWORD *)v29 );
        v10 = v41;
        v43 = (_QWORD *)v50;
      }
      if ( v43 != v52 )
        _libc_free((unsigned __int64)v43);
      if ( v47 == v49 )
      {
LABEL_11:
        v10 = *(_QWORD *)(v10 + 8);
        if ( v40 == v10 )
          return sub_11D2C20(v53);
      }
      else
      {
        _libc_free((unsigned __int64)v47);
        v10 = *(_QWORD *)(v10 + 8);
        if ( v40 == v10 )
          return sub_11D2C20(v53);
      }
    }
  }
  return sub_11D2C20(v53);
}
