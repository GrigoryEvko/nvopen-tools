// Function: sub_F71210
// Address: 0xf71210
//
__int64 __fastcall sub_F71210(unsigned __int64 a1, unsigned __int64 *a2, __int64 a3, unsigned __int64 *a4, __int64 a5)
{
  __int64 *v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // r14
  _QWORD *v16; // rsi
  _QWORD *v17; // r15
  __int64 v18; // rax
  _QWORD *v19; // rsi
  _QWORD *v20; // r15
  __int64 v21; // rax
  __int64 *v22; // rsi
  __int64 result; // rax
  __int64 v24; // r13
  _QWORD *v25; // rbx
  _QWORD *v26; // r12
  __int64 v27; // rax
  _QWORD *v28; // rbx
  _QWORD *v29; // r12
  __int64 v30; // rax
  __int64 v32; // [rsp+10h] [rbp-A0h] BYREF
  __int64 *v33; // [rsp+18h] [rbp-98h] BYREF
  __int64 v34; // [rsp+28h] [rbp-88h] BYREF
  __int64 v35; // [rsp+30h] [rbp-80h] BYREF
  __int64 v36; // [rsp+38h] [rbp-78h] BYREF
  unsigned __int64 *v37[14]; // [rsp+40h] [rbp-70h] BYREF

  v33 = (__int64 *)a1;
  v32 = a5;
  v34 = sub_D47930(a1);
  v35 = **(_QWORD **)(a1 + 32);
  v7 = (__int64 *)a1;
  do
  {
    v8 = v7;
    v7 = (__int64 *)*v7;
  }
  while ( v7 );
  sub_DAC210(a3, a1);
  v9 = 0;
  sub_D9D700(a3, 0);
  v36 = 0;
  if ( v32 )
  {
    v14 = sub_22077B0(760);
    if ( v14 )
    {
      *(_QWORD *)v14 = v32;
      *(_QWORD *)(v14 + 8) = v14 + 24;
      *(_QWORD *)(v14 + 416) = v14 + 440;
      *(_QWORD *)(v14 + 16) = 0x1000000000LL;
      *(_QWORD *)(v14 + 504) = v14 + 520;
      v10 = v14 + 720;
      *(_QWORD *)(v14 + 408) = 0;
      *(_QWORD *)(v14 + 424) = 8;
      *(_DWORD *)(v14 + 432) = 0;
      *(_BYTE *)(v14 + 436) = 1;
      *(_QWORD *)(v14 + 512) = 0x800000000LL;
      *(_DWORD *)(v14 + 720) = 0;
      *(_QWORD *)(v14 + 728) = 0;
      *(_QWORD *)(v14 + 736) = v14 + 720;
      *(_QWORD *)(v14 + 744) = v14 + 720;
      *(_QWORD *)(v14 + 752) = 0;
    }
    v15 = v36;
    v36 = v14;
    if ( v15 )
    {
      sub_F6BD50(*(_QWORD **)(v15 + 728));
      v16 = *(_QWORD **)(v15 + 504);
      v17 = &v16[3 * *(unsigned int *)(v15 + 512)];
      if ( v16 != v17 )
      {
        do
        {
          v18 = *(v17 - 1);
          v17 -= 3;
          if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
            sub_BD60C0(v17);
        }
        while ( v16 != v17 );
        v17 = *(_QWORD **)(v15 + 504);
      }
      if ( v17 != (_QWORD *)(v15 + 520) )
        _libc_free(v17, v16);
      if ( !*(_BYTE *)(v15 + 436) )
        _libc_free(*(_QWORD *)(v15 + 416), v16);
      v19 = *(_QWORD **)(v15 + 8);
      v20 = &v19[3 * *(unsigned int *)(v15 + 16)];
      if ( v19 != v20 )
      {
        do
        {
          v21 = *(v20 - 1);
          v20 -= 3;
          if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
            sub_BD60C0(v20);
        }
        while ( v19 != v20 );
        v20 = *(_QWORD **)(v15 + 8);
      }
      if ( v20 != (_QWORD *)(v15 + 24) )
        _libc_free(v20, v19);
      v9 = 760;
      j_j___libc_free_0(v15, 760);
    }
  }
  v37[6] = a4;
  v37[0] = (unsigned __int64 *)&v34;
  v37[2] = (unsigned __int64 *)&v36;
  v37[3] = (unsigned __int64 *)&v33;
  v37[4] = (unsigned __int64 *)&v35;
  v37[1] = a2;
  v37[5] = (unsigned __int64 *)&v32;
  sub_F70900(v37, v9, v10, v11, v12, v13);
  v22 = v33;
  result = sub_D4F720((__int64)a4, v33);
  if ( v33 != v8 )
  {
    v22 = (__int64 *)a2;
    result = sub_11D2180(v8, a2, a4, a3);
  }
  v24 = v36;
  if ( v36 )
  {
    sub_F6BD50(*(_QWORD **)(v36 + 728));
    v25 = *(_QWORD **)(v24 + 504);
    v26 = &v25[3 * *(unsigned int *)(v24 + 512)];
    if ( v25 != v26 )
    {
      do
      {
        v27 = *(v26 - 1);
        v26 -= 3;
        if ( v27 != -4096 && v27 != 0 && v27 != -8192 )
          sub_BD60C0(v26);
      }
      while ( v25 != v26 );
      v26 = *(_QWORD **)(v24 + 504);
    }
    if ( v26 != (_QWORD *)(v24 + 520) )
      _libc_free(v26, v22);
    if ( !*(_BYTE *)(v24 + 436) )
      _libc_free(*(_QWORD *)(v24 + 416), v22);
    v28 = *(_QWORD **)(v24 + 8);
    v29 = &v28[3 * *(unsigned int *)(v24 + 16)];
    if ( v28 != v29 )
    {
      do
      {
        v30 = *(v29 - 1);
        v29 -= 3;
        if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
          sub_BD60C0(v29);
      }
      while ( v28 != v29 );
      v29 = *(_QWORD **)(v24 + 8);
    }
    if ( v29 != (_QWORD *)(v24 + 24) )
      _libc_free(v29, v22);
    return j_j___libc_free_0(v24, 760);
  }
  return result;
}
