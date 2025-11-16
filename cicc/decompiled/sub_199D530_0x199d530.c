// Function: sub_199D530
// Address: 0x199d530
//
void __fastcall sub_199D530(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __m128i a6, __m128i a7)
{
  __int64 *v10; // rax
  _QWORD *v11; // r15
  __int16 v12; // ax
  __int64 *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  _QWORD **v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // r14
  __int64 v24; // rax
  __int64 *v25; // rax
  _BYTE *v26; // r13
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 *v33; // [rsp+18h] [rbp-108h]
  __int64 *v34; // [rsp+18h] [rbp-108h]
  __int64 v35; // [rsp+20h] [rbp-100h]
  _BYTE *v37; // [rsp+28h] [rbp-F8h]
  __int64 *v38; // [rsp+30h] [rbp-F0h]
  __int64 *v39; // [rsp+30h] [rbp-F0h]
  __int64 v40; // [rsp+30h] [rbp-F0h]
  _QWORD *v41; // [rsp+38h] [rbp-E8h] BYREF
  __int64 *v42; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v43; // [rsp+48h] [rbp-D8h]
  __int64 v44; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v45; // [rsp+58h] [rbp-C8h]
  __int64 *v46[2]; // [rsp+60h] [rbp-C0h] BYREF
  _BYTE v47[32]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 *v48; // [rsp+90h] [rbp-90h] BYREF
  __int64 v49; // [rsp+98h] [rbp-88h]
  _BYTE v50[32]; // [rsp+A0h] [rbp-80h] BYREF
  _BYTE *v51; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v52; // [rsp+C8h] [rbp-58h]
  _BYTE v53[80]; // [rsp+D0h] [rbp-50h] BYREF

  v10 = *(__int64 **)(a2 + 32);
  v41 = a1;
  if ( sub_146D930((__int64)a5, (__int64)a1, *v10) )
  {
    sub_1458920(a3, &v41);
    return;
  }
  v11 = v41;
  v12 = *((_WORD *)v41 + 12);
  if ( v12 == 4 )
  {
    v13 = (__int64 *)v41[4];
    v14 = v41[5];
    if ( v13 != &v13[v14] )
    {
      v38 = &v13[v14];
      do
      {
        v15 = *v13++;
        sub_199D530(v15, a2, a3, a4, a5);
      }
      while ( v38 != v13 );
    }
  }
  else
  {
    if ( v12 == 7 )
    {
      if ( !sub_14560B0(*(_QWORD *)v41[4]) && v41[5] == 2 )
      {
        sub_199D530(*(_QWORD *)v41[4], a2, a3, a4, a5);
        v40 = v41[6];
        v29 = sub_13A5BC0(v41, (__int64)a5);
        v30 = sub_1456040(*(_QWORD *)v41[4]);
        v31 = sub_145CF80((__int64)a5, v30, 0, 0);
        v32 = sub_14799E0((__int64)a5, v31, v29, v40, 0);
        sub_199D530(v32, a2, a3, a4, a5);
        return;
      }
      v11 = v41;
      v12 = *((_WORD *)v41 + 12);
    }
    if ( v12 == 5 && sub_1456170(*(_QWORD *)v11[4]) )
    {
      v16 = v11[4];
      v17 = v11[5];
      v46[1] = (__int64 *)0x400000000LL;
      v46[0] = (__int64 *)v47;
      sub_145C5B0((__int64)v46, (_BYTE *)(v16 + 8), (_BYTE *)(v16 + 8 * v17));
      v18 = sub_147EE30(a5, v46, 0, 0, a6, a7);
      v52 = 0x400000000LL;
      v48 = (__int64 *)v50;
      v49 = 0x400000000LL;
      v51 = v53;
      sub_199D530(v18, a2, &v48, &v51, a5);
      v19 = sub_1456040(v18);
      v20 = (_QWORD **)sub_1456E10((__int64)a5, v19);
      v21 = sub_15A04A0(v20);
      v22 = sub_146F1B0((__int64)a5, v21);
      v23 = v48;
      v35 = v22;
      v33 = &v48[(unsigned int)v49];
      if ( v48 != v33 )
      {
        do
        {
          v24 = *v23;
          v42 = &v44;
          v44 = v35;
          v45 = v24;
          v43 = 0x200000002LL;
          v25 = (__int64 *)sub_147EE30(a5, &v42, 0, 0, a6, a7);
          if ( v42 != &v44 )
          {
            v39 = v25;
            _libc_free((unsigned __int64)v42);
            v25 = v39;
          }
          v42 = v25;
          ++v23;
          sub_1458920(a3, &v42);
        }
        while ( v33 != v23 );
      }
      v26 = v51;
      v37 = &v51[8 * (unsigned int)v52];
      if ( v37 != v51 )
      {
        do
        {
          v27 = *(_QWORD *)v26;
          v43 = 0x200000002LL;
          v42 = &v44;
          v44 = v35;
          v45 = v27;
          v28 = (__int64 *)sub_147EE30(a5, &v42, 0, 0, a6, a7);
          if ( v42 != &v44 )
          {
            v34 = v28;
            _libc_free((unsigned __int64)v42);
            v28 = v34;
          }
          v42 = v28;
          v26 += 8;
          sub_1458920(a4, &v42);
        }
        while ( v37 != v26 );
        v26 = v51;
      }
      if ( v26 != v53 )
        _libc_free((unsigned __int64)v26);
      if ( v48 != (__int64 *)v50 )
        _libc_free((unsigned __int64)v48);
      if ( (_BYTE *)v46[0] != v47 )
        _libc_free((unsigned __int64)v46[0]);
    }
    else
    {
      sub_1458920(a4, &v41);
    }
  }
}
