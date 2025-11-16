// Function: sub_3597E50
// Address: 0x3597e50
//
_QWORD *__fastcall sub_3597E50(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v9; // r10
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  const char *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r10
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // [rsp-8h] [rbp-C8h]
  __int64 v23; // [rsp+8h] [rbp-B8h]
  int v24; // [rsp+10h] [rbp-B0h]
  int v25; // [rsp+18h] [rbp-A8h]
  __int64 v26; // [rsp+20h] [rbp-A0h]
  __int64 v27; // [rsp+20h] [rbp-A0h]
  __int64 v28; // [rsp+28h] [rbp-98h]
  __int64 v29; // [rsp+38h] [rbp-88h]
  __int64 v30; // [rsp+38h] [rbp-88h]
  _QWORD *v31; // [rsp+40h] [rbp-80h] BYREF
  __int64 v32; // [rsp+48h] [rbp-78h]
  _QWORD v33[2]; // [rsp+50h] [rbp-70h] BYREF
  const char *v34; // [rsp+60h] [rbp-60h] BYREF
  __int64 v35; // [rsp+68h] [rbp-58h]
  _QWORD v36[10]; // [rsp+70h] [rbp-50h] BYREF

  v9 = *(_QWORD *)(a2 + 16);
  if ( !v9 )
  {
    if ( qword_503FB90 )
    {
      v34 = (const char *)v36;
      sub_3597AF0((__int64 *)&v34, (_BYTE *)qword_503FB88, qword_503FB88 + qword_503FB90);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v35) <= 2
        || (sub_2241490((unsigned __int64 *)&v34, (char *)&off_3F92B2E, 3u),
            v31 = v33,
            sub_3597AF0((__int64 *)&v31, (_BYTE *)qword_503FB88, qword_503FB88 + qword_503FB90),
            (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v32) <= 3) )
      {
        sub_4262D8((__int64)"basic_string::append");
      }
      v14 = ".out";
      sub_2241490((unsigned __int64 *)&v31, ".out", 4u);
      v23 = sub_B2BE50(*a3);
      v24 = (int)v31;
      v25 = v32;
      v26 = (__int64)v34;
      v28 = v35;
      v15 = sub_22077B0(0xF8u);
      if ( v15 )
      {
        v17 = v26;
        v27 = v15;
        v14 = (const char *)v23;
        sub_36FEDF0(v15, v23, (unsigned int)qword_503FA60, (unsigned int)&unk_503FA80, v24, v25, v17, v28);
        v15 = v27;
        v16 = v22;
      }
      v18 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a2 + 16) = v15;
      if ( v18 )
        (*(void (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v18 + 8LL))(v18, v14, v16);
      if ( v31 != v33 )
        j_j___libc_free_0((unsigned __int64)v31);
      if ( v34 != (const char *)v36 )
        j_j___libc_free_0((unsigned __int64)v34);
      v9 = *(_QWORD *)(a2 + 16);
    }
    else
    {
      v19 = sub_B2BE50(*a3);
      v35 = 5;
      v30 = v19;
      v34 = "feed_";
      v36[0] = "fetch_";
      v36[1] = 6;
      v36[2] = byte_3F871B3;
      v36[3] = 0;
      v20 = sub_22077B0(0x58u);
      v9 = v20;
      if ( v20 )
        sub_3597390(v20, v30, qword_503FA60, (__int64)"priority", 8, (__int64)&v34);
      v21 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a2 + 16) = 0;
      if ( v21 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
        v9 = *(_QWORD *)(a2 + 16);
      }
    }
  }
  v29 = v9;
  v10 = (_QWORD *)sub_22077B0(0x98u);
  v11 = v10;
  if ( v10 )
    sub_3597D20(v10, a3, a4, a5, v29);
  *a1 = v11;
  return a1;
}
