// Function: sub_23AD6C0
// Address: 0x23ad6c0
//
unsigned __int64 *__fastcall sub_23AD6C0(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned __int64 *v9; // rsi
  unsigned __int64 v10; // r14
  __int64 v11; // rbx
  unsigned __int64 *v12; // rsi
  __int64 v14; // rbx
  unsigned __int64 *v15; // rsi
  _BOOL8 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // r9
  unsigned __int64 v23; // r8
  _QWORD *v24; // rax
  unsigned __int64 v25; // r8
  __int64 v26; // [rsp+8h] [rbp-368h]
  char v27; // [rsp+10h] [rbp-360h]
  unsigned __int64 *v28; // [rsp+18h] [rbp-358h]
  unsigned __int64 v29; // [rsp+20h] [rbp-350h]
  unsigned __int64 v30; // [rsp+28h] [rbp-348h]
  unsigned __int64 v31; // [rsp+28h] [rbp-348h]
  unsigned __int64 v32; // [rsp+28h] [rbp-348h]
  unsigned __int64 *v33; // [rsp+28h] [rbp-348h]
  __int64 v34; // [rsp+38h] [rbp-338h] BYREF
  _QWORD *v35; // [rsp+40h] [rbp-330h] BYREF
  _QWORD *v36; // [rsp+48h] [rbp-328h]
  __int64 v37; // [rsp+50h] [rbp-320h]
  unsigned __int64 v38; // [rsp+58h] [rbp-318h]
  unsigned __int64 *v39; // [rsp+60h] [rbp-310h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  if ( a4 )
  {
    if ( LOBYTE(qword_4FF33E0[17]) )
    {
      v16 = 0;
      if ( *(_BYTE *)(a2 + 192) )
        v16 = *(_DWORD *)(a2 + 168) == 3;
      sub_264AF10(&v35, a4, v16);
      v17 = (unsigned __int64)v36;
      v36 = 0;
      v27 = v37;
      v28 = v39;
      v26 = (__int64)v35;
      v32 = v17;
      v29 = v38;
      v38 = 0;
      v39 = 0;
      v18 = sub_22077B0(0x30u);
      if ( v18 )
      {
        *(_QWORD *)(v18 + 40) = v28;
        v34 = v18;
        *(_QWORD *)(v18 + 8) = v26;
        *(_BYTE *)(v18 + 24) = v27;
        *(_QWORD *)v18 = &unk_4A0D878;
        v19 = (__int64)&v34;
        *(_QWORD *)(v18 + 16) = v32;
        *(_QWORD *)(v18 + 32) = v29;
        sub_23A2230(a1, (unsigned __int64 *)&v34);
        sub_23501E0(&v34);
      }
      else
      {
        v19 = (__int64)&v34;
        v34 = 0;
        sub_23A2230(a1, (unsigned __int64 *)&v34);
        sub_23501E0(&v34);
        v25 = (unsigned __int64)v28;
        if ( v28 )
        {
          if ( (unsigned __int64 *)*v28 != v28 + 2 )
          {
            _libc_free(*v28);
            v25 = (unsigned __int64)v28;
          }
          v19 = 80;
          j_j___libc_free_0(v25);
        }
        if ( v29 )
          sub_23A1CE0(v29, v19, v20, v21, v25, v22);
        if ( v32 )
          sub_23A0670(v32);
      }
      v23 = (unsigned __int64)v39;
      if ( v39 )
      {
        if ( (unsigned __int64 *)*v39 != v39 + 2 )
        {
          v33 = v39;
          _libc_free(*v39);
          v23 = (unsigned __int64)v33;
        }
        v19 = 80;
        j_j___libc_free_0(v23);
      }
      if ( v38 )
        sub_23A1CE0(v38, v19, v20, v21, v23, v22);
      if ( v36 )
        sub_23A0670((unsigned __int64)v36);
    }
    v7 = sub_22077B0(0x20u);
    if ( v7 )
    {
      *(_QWORD *)(v7 + 8) = 0;
      *(_QWORD *)(v7 + 16) = a4;
      *(_BYTE *)(v7 + 24) = 0;
      *(_QWORD *)v7 = &unk_4A0E478;
    }
    v35 = (_QWORD *)v7;
    sub_23A2230(a1, (unsigned __int64 *)&v35);
    sub_23501E0((__int64 *)&v35);
    LOBYTE(v35) = 0;
    v36 = 0;
    v37 = a4;
    LODWORD(v38) = 0;
    sub_23A23F0(a1, (char *)&v35);
  }
  if ( __PAIR64__(HIDWORD(qword_5033F08), qword_5033F08) == a3 )
  {
    v35 = 0;
    v36 = 0;
    v37 = 0;
    LODWORD(v38) = 1;
    sub_23A23F0(a1, (char *)&v35);
    v24 = (_QWORD *)sub_22077B0(0x10u);
    if ( v24 )
      *v24 = &unk_4A0D238;
    v35 = v24;
    sub_23A2230(a1, (unsigned __int64 *)&v35);
    sub_23501E0((__int64 *)&v35);
    sub_23A0BA0((__int64)&v35, 0);
    sub_23A2670(a1, (__int64)&v35);
    sub_233AAF0((__int64)&v35);
  }
  else
  {
    if ( qword_502E468[9] )
    {
      sub_23A7230((unsigned __int64 *)&v35, a2, a3, 2);
      v8 = (__int64)v35;
      v30 = (unsigned __int64)v36;
      if ( v35 != v36 )
      {
        do
        {
          v9 = (unsigned __int64 *)v8;
          v8 += 8;
          sub_23A2230(a1, v9);
        }
        while ( v30 != v8 );
      }
    }
    else
    {
      sub_23A7740((unsigned __int64 *)&v35, a2, a3, 2);
      v14 = (__int64)v35;
      v31 = (unsigned __int64)v36;
      if ( v35 != v36 )
      {
        do
        {
          v15 = (unsigned __int64 *)v14;
          v14 += 8;
          sub_23A2230(a1, v15);
        }
        while ( v31 != v14 );
      }
    }
    sub_234A900((__int64)&v35);
    sub_23AC3F0((unsigned __int64 *)&v35, a2, a3, 2);
    v10 = (unsigned __int64)v36;
    v11 = (__int64)v35;
    if ( v35 != v36 )
    {
      do
      {
        v12 = (unsigned __int64 *)v11;
        v11 += 8;
        sub_23A2230(a1, v12);
      }
      while ( v10 != v11 );
    }
    sub_234A900((__int64)&v35);
    sub_23A2610(a1);
  }
  return a1;
}
