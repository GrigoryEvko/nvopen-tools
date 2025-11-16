// Function: sub_1555EB0
// Address: 0x1555eb0
//
__int64 __fastcall sub_1555EB0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  void *v18; // [rsp+10h] [rbp-130h] BYREF
  __int64 v19; // [rsp+18h] [rbp-128h]
  __int64 v20; // [rsp+20h] [rbp-120h]
  __int64 v21; // [rsp+28h] [rbp-118h]
  int v22; // [rsp+30h] [rbp-110h]
  __int64 v23; // [rsp+38h] [rbp-108h]
  _QWORD v24[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v25; // [rsp+60h] [rbp-E0h]
  __int64 v26; // [rsp+68h] [rbp-D8h]
  __int64 v27; // [rsp+70h] [rbp-D0h]
  __int64 v28; // [rsp+78h] [rbp-C8h]
  __int64 v29; // [rsp+80h] [rbp-C0h]
  __int64 v30; // [rsp+88h] [rbp-B8h]
  __int64 v31; // [rsp+90h] [rbp-B0h]
  __int64 v32; // [rsp+98h] [rbp-A8h]
  __int64 v33; // [rsp+A0h] [rbp-A0h]
  __int64 v34; // [rsp+A8h] [rbp-98h]
  __int64 v35; // [rsp+B0h] [rbp-90h]
  __int64 v36; // [rsp+B8h] [rbp-88h]
  __int64 v37; // [rsp+C0h] [rbp-80h]
  __int64 v38; // [rsp+C8h] [rbp-78h]
  char v39; // [rsp+D0h] [rbp-70h]
  __int64 v40; // [rsp+D8h] [rbp-68h]
  __int64 v41; // [rsp+E0h] [rbp-60h]
  __int64 v42; // [rsp+E8h] [rbp-58h]
  int v43; // [rsp+F0h] [rbp-50h]
  __int64 v44; // [rsp+F8h] [rbp-48h]
  __int64 v45; // [rsp+100h] [rbp-40h]
  __int64 v46; // [rsp+108h] [rbp-38h]

  sub_154B550((__int64)&v18, a1);
  v24[0] = a4;
  v24[1] = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v7 = sub_154BC70(a3);
  sub_154F770((__int64)&v18, a2, (__int64)v24, v7, a4);
  if ( (unsigned __int8)(*a2 - 4) > 0x1Eu || *a2 == 6 || a5 )
  {
    if ( v44 )
      j_j___libc_free_0(v44, v46 - v44);
    j___libc_free_0(v41);
    if ( v36 )
      j_j___libc_free_0(v36, v38 - v36);
    j___libc_free_0(v33);
    j___libc_free_0(v29);
    j___libc_free_0(v25);
    v18 = &unk_49EF340;
    if ( v21 != v19 )
      sub_16E7BA0(&v18);
    v8 = v23;
    if ( v23 )
    {
      if ( !v22 || v19 )
      {
        v9 = v20 - v19;
      }
      else
      {
        v16 = sub_16E7720(&v18);
        v8 = v23;
        v9 = v16;
      }
      v10 = *(_QWORD *)(v8 + 24);
      v11 = *(_QWORD *)(v8 + 8);
      if ( !v9 )
        goto LABEL_30;
LABEL_12:
      if ( v10 != v11 )
        sub_16E7BA0(v8);
      v12 = sub_2207820(v9);
      sub_16E7A40(v8, v12, v9, 1);
    }
  }
  else
  {
    sub_1263B40((__int64)&v18, " = ");
    v14 = sub_154BC70(a3);
    sub_1553C10((__int64)&v18, (__int64)a2, (__int64)v24, v14, a4);
    if ( v44 )
      j_j___libc_free_0(v44, v46 - v44);
    j___libc_free_0(v41);
    if ( v36 )
      j_j___libc_free_0(v36, v38 - v36);
    j___libc_free_0(v33);
    j___libc_free_0(v29);
    j___libc_free_0(v25);
    v18 = &unk_49EF340;
    if ( v21 != v19 )
      sub_16E7BA0(&v18);
    v8 = v23;
    if ( v23 )
    {
      if ( !v22 || v19 )
      {
        v9 = v20 - v19;
      }
      else
      {
        v15 = sub_16E7720(&v18);
        v8 = v23;
        v9 = v15;
      }
      v11 = *(_QWORD *)(v8 + 24);
      v10 = *(_QWORD *)(v8 + 8);
      if ( !v9 )
      {
LABEL_30:
        if ( v10 != v11 )
          sub_16E7BA0(v8);
        sub_16E7A40(v8, 0, 0, 0);
        return sub_16E7960(&v18);
      }
      goto LABEL_12;
    }
  }
  return sub_16E7960(&v18);
}
