// Function: sub_155BD40
// Address: 0x155bd40
//
__int64 __fastcall sub_155BD40(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rax
  _BYTE *v6; // rdx
  unsigned __int8 v7; // al
  __int64 v8; // rax
  __int64 v9; // rsi
  _QWORD *v10; // rax
  unsigned __int64 v11; // r8
  __int64 v12; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  _QWORD *v23; // rax
  char v24; // al
  _QWORD *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  _BYTE *v28; // [rsp+0h] [rbp-420h]
  __int64 v29; // [rsp+0h] [rbp-420h]
  _BYTE *v30; // [rsp+0h] [rbp-420h]
  __int64 v31; // [rsp+0h] [rbp-420h]
  void *v33; // [rsp+10h] [rbp-410h] BYREF
  __int64 v34; // [rsp+18h] [rbp-408h]
  __int64 v35; // [rsp+20h] [rbp-400h]
  __int64 v36; // [rsp+28h] [rbp-3F8h]
  int v37; // [rsp+30h] [rbp-3F0h]
  __int64 v38; // [rsp+38h] [rbp-3E8h]
  _BYTE v39[40]; // [rsp+50h] [rbp-3D0h] BYREF
  __int64 v40; // [rsp+78h] [rbp-3A8h]
  __int64 v41; // [rsp+A0h] [rbp-380h]
  __int64 v42; // [rsp+C8h] [rbp-358h]
  __int64 v43; // [rsp+F0h] [rbp-330h]
  unsigned __int64 v44; // [rsp+110h] [rbp-310h]
  unsigned int v45; // [rsp+118h] [rbp-308h]
  int v46; // [rsp+11Ch] [rbp-304h]
  __int64 v47; // [rsp+140h] [rbp-2E0h]
  __int64 v48[2]; // [rsp+160h] [rbp-2C0h] BYREF
  __int64 v49; // [rsp+170h] [rbp-2B0h]
  __int64 v50; // [rsp+178h] [rbp-2A8h]
  __int64 v51; // [rsp+180h] [rbp-2A0h]
  __int64 v52; // [rsp+188h] [rbp-298h]
  __int64 v53; // [rsp+190h] [rbp-290h]
  __int64 v54; // [rsp+198h] [rbp-288h]
  __int64 v55; // [rsp+1A0h] [rbp-280h]
  __int64 v56; // [rsp+1A8h] [rbp-278h]
  __int64 v57; // [rsp+1B0h] [rbp-270h]
  __int64 v58; // [rsp+1B8h] [rbp-268h]
  __int64 v59; // [rsp+1C0h] [rbp-260h]
  __int64 v60; // [rsp+1C8h] [rbp-258h]
  __int64 v61; // [rsp+1D0h] [rbp-250h]
  __int64 v62; // [rsp+1D8h] [rbp-248h]
  __int64 v63; // [rsp+1E0h] [rbp-240h]
  __int64 v64; // [rsp+1E8h] [rbp-238h]
  __int64 v65; // [rsp+1F0h] [rbp-230h]
  __int64 v66; // [rsp+1F8h] [rbp-228h]
  __int64 v67; // [rsp+200h] [rbp-220h]
  __int64 v68; // [rsp+208h] [rbp-218h]
  __int64 v69; // [rsp+210h] [rbp-210h]
  __int64 v70; // [rsp+218h] [rbp-208h]

  sub_154B550((__int64)&v33, a2);
  sub_154BB30((__int64)v39, 0, 0);
  v5 = sub_154BC70(a3);
  v6 = v39;
  if ( v5 )
    v6 = (_BYTE *)sub_154BC70(a3);
  v7 = *(_BYTE *)(a1 + 16);
  if ( v7 <= 0x17u )
  {
    if ( v7 == 18 )
    {
      v22 = *(_QWORD *)(a1 + 56);
      if ( v22 )
      {
        v30 = v6;
        sub_154C150(a3, v22);
        v6 = v30;
      }
      v31 = (__int64)v6;
      v23 = sub_1548BC0(a1);
      sub_1556670((__int64)v48, (__int64)&v33, v31, (__int64)v23, 0, a4, 0);
      sub_1558F20(v48, a1);
      sub_1549650(v48);
    }
    else if ( v7 > 3u )
    {
      if ( v7 == 19 )
      {
        v25 = sub_1548BC0(a1);
        sub_15562E0(*(unsigned __int8 **)(a1 + 24), a2, a3, (__int64)v25);
      }
      else if ( v7 > 0x10u )
      {
        sub_1553920((__int64 *)a1, (__int64)&v33, 1, a3);
      }
      else
      {
        v26 = *(_QWORD *)a1;
        v48[0] = 0;
        v48[1] = 0;
        v49 = 0;
        v50 = 0;
        v51 = 0;
        v52 = 0;
        v53 = 0;
        v54 = 0;
        v55 = 0;
        v56 = 0;
        v57 = 0;
        v58 = 0;
        v59 = 0;
        v60 = 0;
        v61 = 0;
        v62 = 0;
        v63 = 0;
        v64 = 0;
        v65 = 0;
        v66 = 0;
        v67 = 0;
        v68 = 0;
        v69 = 0;
        v70 = 0;
        sub_154DAA0((__int64)v48, v26, (__int64)&v33);
        sub_1549FC0((__int64)&v33, 0x20u);
        v27 = sub_154BC70(a3);
        sub_15510D0((__int64)&v33, (__int64 *)a1, (__int64)v48, v27, 0);
        if ( v68 )
          j_j___libc_free_0(v68, v70 - v68);
        j___libc_free_0(v65);
        if ( v60 )
          j_j___libc_free_0(v60, v62 - v60);
        j___libc_free_0(v57);
        j___libc_free_0(v53);
        j___libc_free_0(v49);
      }
    }
    else
    {
      sub_1556670((__int64)v48, (__int64)&v33, (__int64)v6, *(_QWORD *)(a1 + 40), 0, a4, 0);
      v24 = *(_BYTE *)(a1 + 16);
      if ( v24 == 3 )
      {
        sub_1552C10(v48, a1);
      }
      else if ( v24 )
      {
        sub_1553050(v48, a1);
      }
      else
      {
        sub_1559290(v48, a1);
      }
      sub_1549650(v48);
    }
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 40);
    if ( v8 )
    {
      v9 = *(_QWORD *)(v8 + 56);
      if ( v9 )
      {
        v28 = v6;
        sub_154C150(a3, v9);
        v6 = v28;
      }
    }
    v29 = (__int64)v6;
    v10 = sub_1548BC0(a1);
    sub_1556670((__int64)v48, (__int64)&v33, v29, (__int64)v10, 0, a4, 0);
    sub_15572A0(v48, a1);
    sub_1549650(v48);
  }
  j___libc_free_0(v47);
  if ( v46 )
  {
    v11 = v44;
    if ( v45 )
    {
      v12 = 8LL * v45;
      v13 = 0;
      do
      {
        v14 = *(_QWORD *)(v11 + v13);
        if ( v14 != -8 && v14 )
        {
          _libc_free(v14);
          v11 = v44;
        }
        v13 += 8;
      }
      while ( v12 != v13 );
    }
  }
  else
  {
    v11 = v44;
  }
  _libc_free(v11);
  j___libc_free_0(v43);
  j___libc_free_0(v42);
  j___libc_free_0(v41);
  j___libc_free_0(v40);
  v33 = &unk_49EF340;
  if ( v36 != v34 )
    sub_16E7BA0(&v33);
  v15 = v38;
  if ( v38 )
  {
    if ( !v37 || v34 )
    {
      v16 = *(_QWORD *)(v38 + 24);
      v17 = v35 - v34;
      v18 = *(_QWORD *)(v38 + 8);
      if ( v35 == v34 )
      {
LABEL_27:
        if ( v16 != v18 )
          sub_16E7BA0(v15);
        sub_16E7A40(v15, 0, 0, 0);
        return sub_16E7960(&v33);
      }
    }
    else
    {
      v21 = sub_16E7720(&v33);
      v15 = v38;
      v17 = v21;
      v16 = *(_QWORD *)(v38 + 24);
      v18 = *(_QWORD *)(v38 + 8);
      if ( !v17 )
        goto LABEL_27;
    }
    if ( v16 != v18 )
      sub_16E7BA0(v15);
    v19 = sub_2207820(v17);
    sub_16E7A40(v15, v19, v17, 1);
  }
  return sub_16E7960(&v33);
}
