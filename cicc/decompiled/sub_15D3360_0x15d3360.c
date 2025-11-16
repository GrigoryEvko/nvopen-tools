// Function: sub_15D3360
// Address: 0x15d3360
//
__int64 __fastcall sub_15D3360(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  __int64 v5; // rax
  char **v6; // rax
  __int64 v7; // rax
  int v8; // eax
  char *v9; // rdx
  __int64 *v10; // rax
  char **v11; // r13
  char *v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // r12
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 *v17; // rbx
  _QWORD *v18; // rbx
  _QWORD *v19; // r12
  unsigned __int64 v20; // rdi
  _QWORD *v22; // rbx
  _QWORD *v23; // r12
  unsigned __int64 v24; // rdi
  int v26; // [rsp+4Ch] [rbp-314h]
  char **v27; // [rsp+50h] [rbp-310h]
  __int64 *v28; // [rsp+58h] [rbp-308h]
  __int64 v29; // [rsp+68h] [rbp-2F8h] BYREF
  char *v30; // [rsp+70h] [rbp-2F0h] BYREF
  char *v31; // [rsp+78h] [rbp-2E8h] BYREF
  char *v32; // [rsp+80h] [rbp-2E0h] BYREF
  __int64 v33; // [rsp+88h] [rbp-2D8h] BYREF
  _QWORD *v34; // [rsp+90h] [rbp-2D0h] BYREF
  _QWORD *v35; // [rsp+98h] [rbp-2C8h]
  _QWORD *v36; // [rsp+A0h] [rbp-2C0h]
  __int64 v37; // [rsp+A8h] [rbp-2B8h] BYREF
  _QWORD *v38; // [rsp+B0h] [rbp-2B0h]
  __int64 v39; // [rsp+B8h] [rbp-2A8h]
  unsigned int v40; // [rsp+C0h] [rbp-2A0h]
  __int64 v41; // [rsp+C8h] [rbp-298h]
  char **v42; // [rsp+D0h] [rbp-290h] BYREF
  int v43; // [rsp+D8h] [rbp-288h]
  char v44; // [rsp+E0h] [rbp-280h] BYREF
  char *v45; // [rsp+120h] [rbp-240h] BYREF
  __int64 v46; // [rsp+128h] [rbp-238h]
  _QWORD v47[70]; // [rsp+130h] [rbp-230h] BYREF

  v2 = *(_QWORD *)(a1 + 64);
  sub_15CE0F0(a1 + 24);
  *(_QWORD *)(a1 + 64) = v2;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_BYTE *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 76) = 0;
  v35 = 0;
  v36 = 0;
  v3 = (_QWORD *)sub_22077B0(8);
  v45 = (char *)v47;
  v4 = v3 + 1;
  v34 = v3;
  *v3 = 0;
  v46 = 0x100000000LL;
  v5 = *(_QWORD *)(a1 + 64);
  v36 = v4;
  v6 = *(char ***)(v5 + 80);
  v35 = v4;
  v37 = 0;
  v38 = 0;
  if ( v6 )
    v6 -= 3;
  v39 = 0;
  v40 = 0;
  v42 = v6;
  v41 = 0;
  sub_15CDD90((__int64)&v45, &v42);
  sub_15CBF70(a1, &v45);
  if ( v45 != (char *)v47 )
    _libc_free((unsigned __int64)v45);
  v7 = **(_QWORD **)a1;
  v45 = (char *)v47;
  v46 = 0x4000000001LL;
  v29 = v7;
  v47[0] = v7;
  v33 = v7;
  if ( (unsigned __int8)sub_15CE630((__int64)&v37, &v33, &v42) )
    *((_DWORD *)sub_15D1D60((__int64)&v37, &v29) + 3) = 0;
  v8 = v46;
  v26 = 0;
  if ( (_DWORD)v46 )
  {
LABEL_10:
    while ( 1 )
    {
      v9 = *(char **)&v45[8 * v8 - 8];
      LODWORD(v46) = v8 - 1;
      v30 = v9;
      v10 = sub_15D1D60((__int64)&v37, (__int64 *)&v30);
      if ( !*((_DWORD *)v10 + 2) )
        break;
LABEL_9:
      v8 = v46;
      if ( !(_DWORD)v46 )
        goto LABEL_23;
    }
    ++v26;
    v10[3] = (__int64)v30;
    *((_DWORD *)v10 + 4) = v26;
    *((_DWORD *)v10 + 2) = v26;
    sub_15CE600((__int64)&v34, &v30);
    sub_15CF6C0((__int64)&v42, (__int64)v30, v41);
    v27 = &v42[v43];
    if ( v42 == v27 )
      goto LABEL_21;
    v11 = v42;
    while ( 1 )
    {
      while ( 1 )
      {
        v31 = *v11;
        v32 = v31;
        if ( (unsigned __int8)sub_15CE630((__int64)&v37, (__int64 *)&v32, &v33) )
        {
          if ( (_QWORD *)v33 != &v38[9 * v40] && *(_DWORD *)(v33 + 8) )
            break;
        }
        v28 = sub_15D1D60((__int64)&v37, (__int64 *)&v31);
        sub_15CDD90((__int64)&v45, &v31);
        *((_DWORD *)v28 + 3) = v26;
        sub_15CDD90((__int64)(v28 + 5), &v30);
LABEL_14:
        if ( v27 == ++v11 )
          goto LABEL_20;
      }
      if ( v31 == v30 )
        goto LABEL_14;
      ++v11;
      sub_15CDD90(v33 + 40, &v30);
      if ( v27 == v11 )
      {
LABEL_20:
        v27 = v42;
LABEL_21:
        if ( v27 == (char **)&v44 )
          goto LABEL_9;
        _libc_free((unsigned __int64)v27);
        v8 = v46;
        if ( !(_DWORD)v46 )
          break;
        goto LABEL_10;
      }
    }
  }
LABEL_23:
  if ( v45 != (char *)v47 )
    _libc_free((unsigned __int64)v45);
  sub_15D2F60((__int64 *)&v34, a1, 0);
  if ( a2 )
    *(_BYTE *)(a2 + 144) = 1;
  if ( *(_DWORD *)(a1 + 8) )
  {
    v12 = **(char ***)a1;
    v45 = v12;
    v13 = sub_22077B0(56);
    v14 = (__int64 *)v13;
    if ( v13 )
    {
      *(_QWORD *)v13 = v12;
      *(_QWORD *)(v13 + 8) = 0;
      *(_DWORD *)(v13 + 16) = 0;
      *(_QWORD *)(v13 + 24) = 0;
      *(_QWORD *)(v13 + 32) = 0;
      *(_QWORD *)(v13 + 40) = 0;
      *(_QWORD *)(v13 + 48) = -1;
    }
    v15 = sub_15CFF10(a1 + 24, (__int64 *)&v45);
    v16 = v15[1];
    v17 = v15;
    v15[1] = (__int64)v14;
    if ( v16 )
    {
      sub_15CBC60(v16);
      v14 = (__int64 *)v17[1];
    }
    *(_QWORD *)(a1 + 56) = v14;
    sub_15D2160((__int64)&v34, a1, v14);
    if ( v40 )
    {
      v18 = v38;
      v19 = &v38[9 * v40];
      do
      {
        if ( *v18 != -16 && *v18 != -8 )
        {
          v20 = v18[5];
          if ( (_QWORD *)v20 != v18 + 7 )
            _libc_free(v20);
        }
        v18 += 9;
      }
      while ( v19 != v18 );
    }
  }
  else if ( v40 )
  {
    v22 = v38;
    v23 = &v38[9 * v40];
    do
    {
      if ( *v22 != -16 && *v22 != -8 )
      {
        v24 = v22[5];
        if ( (_QWORD *)v24 != v22 + 7 )
          _libc_free(v24);
      }
      v22 += 9;
    }
    while ( v23 != v22 );
  }
  j___libc_free_0(v38);
  return sub_15CE080(&v34);
}
