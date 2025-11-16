// Function: sub_B8B080
// Address: 0xb8b080
//
__int64 __fastcall sub_B8B080(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 v5; // rax
  __int64 v6; // rbx
  _QWORD *v7; // r15
  __int64 v8; // r13
  _QWORD *v9; // rax
  __int64 result; // rax
  void (__fastcall *v11)(__int64 *, __int64, _QWORD); // r15
  unsigned int v12; // eax
  unsigned __int8 *v13; // rax
  __int64 v14; // r14
  __int64 (__fastcall *v15)(__int64, __int64, _QWORD); // rbx
  unsigned int v16; // eax
  int v17; // eax
  __int64 v18; // rsi
  int v19; // ecx
  unsigned int v20; // edx
  __int64 **v21; // rax
  __int64 *v22; // rdi
  _QWORD *v23; // rdi
  unsigned __int64 v24; // rdx
  _BYTE *v25; // r11
  size_t v26; // r9
  unsigned __int64 v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // r15
  void (__fastcall *v31)(__int64 *, __int64, _QWORD); // r14
  unsigned int v32; // eax
  void (__fastcall *v33)(__int64 *, __int64, _QWORD); // r15
  unsigned int v34; // eax
  __int64 v35; // rax
  int v36; // eax
  int v37; // r9d
  size_t n; // [rsp+10h] [rbp-80h]
  _BYTE *src; // [rsp+18h] [rbp-78h]
  __int64 (__fastcall *v40)(__int64 *, __int64, _QWORD **); // [rsp+20h] [rbp-70h]
  __int64 v41; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v42; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v43; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v44; // [rsp+48h] [rbp-48h]
  _QWORD v45[8]; // [rsp+50h] [rbp-40h] BYREF

  v3 = a1 + 8;
  (*(void (__fastcall **)(__int64 *, __int64))(*a2 + 72))(a2, a1 + 8);
  v5 = sub_B85AD0(a1, a2[2]);
  v6 = v5;
  if ( !v5 )
  {
    v41 = a1;
    sub_B8A8E0(&v41, a2);
    v7 = (_QWORD *)(*(__int64 (__fastcall **)(__int64 *))(*a2 + 112))(a2);
    if ( v7 )
      goto LABEL_4;
    v33 = *(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a2 + 64);
    v34 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 8LL))(a1);
    v33(a2, v3, v34);
    sub_BC5D20();
    goto LABEL_13;
  }
  if ( !*(_BYTE *)(v5 + 41) || !sub_B811E0(a1, a2[2]) )
  {
    v41 = a1;
    sub_B8A8E0(&v41, a2);
    v7 = (_QWORD *)(*(__int64 (__fastcall **)(__int64 *))(*a2 + 112))(a2);
    if ( v7 )
    {
LABEL_4:
      v8 = (**(__int64 (__fastcall ***)(__int64))a1)(a1);
      v9 = (_QWORD *)sub_22077B0(32);
      if ( v9 )
      {
        *v9 = 0;
        v9[1] = 0;
        v9[2] = 0;
        v9[3] = v8;
      }
      sub_BB9580(a2, v9);
      sub_B89740(v8, a2);
      sub_B86C20(a1, v7);
      return (__int64)sub_B87180(v8, (__int64)v7);
    }
    if ( !*(_BYTE *)(v6 + 41) )
      sub_BC5D40(*(_QWORD *)(v6 + 16), *(_QWORD *)(v6 + 24));
    v11 = *(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a2 + 64);
    v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 8LL))(a1);
    v11(a2, v3, v12);
    if ( !*(_BYTE *)(v6 + 41) )
      sub_BC5D90(*(_QWORD *)(v6 + 16), *(_QWORD *)(v6 + 24));
    if ( (unsigned __int8)sub_BC5D20() && !*(_BYTE *)(v6 + 41) )
    {
      v23 = a2;
      v40 = *(__int64 (__fastcall **)(__int64 *, __int64, _QWORD **))(*a2 + 56);
      v25 = (_BYTE *)(*(__int64 (__fastcall **)(__int64 *))(*a2 + 16))(a2);
      v26 = v24;
      if ( !v25 )
      {
        v44 = 0;
        v43 = v45;
        LOBYTE(v45[0]) = 0;
        goto LABEL_30;
      }
      v42 = v24;
      v27 = v24;
      v43 = v45;
      if ( v24 > 0xF )
      {
        n = v24;
        src = v25;
        v35 = sub_22409D0(&v43, &v42, 0);
        v25 = src;
        v26 = n;
        v43 = (_QWORD *)v35;
        v23 = (_QWORD *)v35;
        v45[0] = v42;
      }
      else
      {
        if ( v24 == 1 )
        {
          LOBYTE(v45[0]) = *v25;
          v28 = v45;
LABEL_29:
          v44 = v27;
          *((_BYTE *)v28 + v27) = 0;
LABEL_30:
          v29 = sub_C5F790(v23);
          v30 = (__int64 *)v40(a2, v29, &v43);
          if ( v43 != v45 )
            j_j___libc_free_0(v43, v45[0] + 1LL);
          if ( v30 )
          {
            sub_B8A8E0(&v41, v30);
            v31 = *(void (__fastcall **)(__int64 *, __int64, _QWORD))(*v30 + 64);
            v32 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 8LL))(a1);
            v31(v30, v3, v32);
          }
          goto LABEL_13;
        }
        if ( !v24 )
        {
          v28 = v45;
          goto LABEL_29;
        }
        v23 = v45;
      }
      memcpy(v23, v25, v26);
      v27 = v42;
      v28 = v43;
      goto LABEL_29;
    }
LABEL_13:
    v13 = (unsigned __int8 *)sub_C94E20(qword_4F86270);
    if ( v13 )
      result = *v13;
    else
      result = LOBYTE(qword_4F86270[2]);
    if ( v6 && (_BYTE)result && !*(_BYTE *)(v6 + 41) )
    {
      v14 = sub_BE0980(1);
      v15 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v14 + 64LL);
      v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 8LL))(a1);
      return v15(v14, v3, v16);
    }
    return result;
  }
  v17 = *(_DWORD *)(a1 + 680);
  v18 = *(_QWORD *)(a1 + 664);
  if ( v17 )
  {
    v19 = v17 - 1;
    v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v21 = (__int64 **)(v18 + 16LL * v20);
    v22 = *v21;
    if ( *v21 == a2 )
    {
LABEL_22:
      *v21 = (__int64 *)-8192LL;
      --*(_DWORD *)(a1 + 672);
      ++*(_DWORD *)(a1 + 676);
    }
    else
    {
      v36 = 1;
      while ( v22 != (__int64 *)-4096LL )
      {
        v37 = v36 + 1;
        v20 = v19 & (v36 + v20);
        v21 = (__int64 **)(v18 + 16LL * v20);
        v22 = *v21;
        if ( *v21 == a2 )
          goto LABEL_22;
        v36 = v37;
      }
    }
  }
  return (*(__int64 (__fastcall **)(__int64 *, __int64))(*a2 + 8))(a2, v18);
}
