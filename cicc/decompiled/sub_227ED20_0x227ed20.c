// Function: sub_227ED20
// Address: 0x227ed20
//
__int64 __fastcall sub_227ED20(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // r15
  bool v6; // zf
  __int64 *v7; // rax
  __int64 v8; // rax
  unsigned int v10; // esi
  int v11; // ebx
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned int v15; // edx
  void **v16; // rax
  __int64 *v17; // r8
  void *v18; // r12
  __int64 v19; // rax
  _QWORD *v20; // rbx
  _QWORD *v21; // rax
  _QWORD *v22; // r15
  __int64 *v23; // rax
  __int64 v24; // rdi
  _QWORD *v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 *v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rax
  _QWORD *v31; // rbx
  _QWORD *v32; // r15
  __int64 *v33; // rax
  __int64 v34; // rdi
  _QWORD *v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rdx
  int v39; // esi
  int v40; // edx
  unsigned int v41; // esi
  __int64 *v42; // rdx
  __int64 v43; // rbx
  int v44; // eax
  int v45; // r9d
  __int64 v47; // [rsp+18h] [rbp-88h]
  __int64 *v48; // [rsp+20h] [rbp-80h]
  __int64 v49; // [rsp+28h] [rbp-78h]
  __int64 v51; // [rsp+30h] [rbp-70h]
  __int64 *v53; // [rsp+40h] [rbp-60h] BYREF
  __int64 *v54; // [rsp+48h] [rbp-58h] BYREF
  __int64 *v55; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v56; // [rsp+58h] [rbp-48h]
  __int64 v57; // [rsp+60h] [rbp-40h]

  v4 = a1;
  v55 = a2;
  v56 = a3;
  v49 = a1 + 64;
  v57 = 0;
  v6 = (unsigned __int8)sub_227C290(a1 + 64, (__int64 *)&v55, &v53) == 0;
  v7 = v53;
  if ( !v6 )
  {
    v8 = v53[2];
    return *(_QWORD *)(v8 + 24);
  }
  v10 = *(_DWORD *)(a1 + 88);
  v11 = *(_DWORD *)(a1 + 80);
  v54 = v53;
  ++*(_QWORD *)(a1 + 64);
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v10 )
  {
    v43 = a1 + 64;
    sub_227EA50(v49, 2 * v10);
LABEL_54:
    sub_227C290(v43, (__int64 *)&v55, &v54);
    v12 = *(_DWORD *)(a1 + 80) + 1;
    v7 = v54;
    goto LABEL_6;
  }
  if ( v10 - *(_DWORD *)(a1 + 84) - v12 <= v10 >> 3 )
  {
    v43 = a1 + 64;
    sub_227EA50(v49, v10);
    goto LABEL_54;
  }
LABEL_6:
  *(_DWORD *)(a1 + 80) = v12;
  if ( *v7 != -4096 || v7[1] != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v7 = (__int64)v55;
  v7[1] = (__int64)v56;
  v7[2] = v57;
  v13 = *(unsigned int *)(a1 + 24);
  v14 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)v13 )
  {
LABEL_50:
    v16 = (void **)(v14 + 16 * v13);
    goto LABEL_10;
  }
  v15 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v16 = (void **)(v14 + 16LL * v15);
  v17 = (__int64 *)*v16;
  if ( *v16 != a2 )
  {
    v44 = 1;
    while ( v17 != (__int64 *)-4096LL )
    {
      v45 = v44 + 1;
      v15 = (v13 - 1) & (v44 + v15);
      v16 = (void **)(v14 + 16LL * v15);
      v17 = (__int64 *)*v16;
      if ( *v16 == a2 )
        goto LABEL_10;
      v44 = v45;
    }
    goto LABEL_50;
  }
LABEL_10:
  v18 = v16[1];
  if ( a2 == (__int64 *)&unk_4F8A320 || (v19 = *(_QWORD *)(sub_227ED20(a1, &unk_4F8A320, a3, a4) + 8), (v47 = v19) == 0) )
  {
    v47 = 0;
  }
  else
  {
    v20 = *(_QWORD **)(v19 + 720);
    v21 = &v20[4 * *(unsigned int *)(v19 + 728)];
    if ( v20 != v21 )
    {
      v22 = v21;
      do
      {
        v55 = 0;
        v23 = (__int64 *)sub_22077B0(0x10u);
        if ( v23 )
        {
          v23[1] = (__int64)a3;
          *v23 = (__int64)&unk_4A08BA8;
        }
        v24 = (__int64)v55;
        v55 = v23;
        if ( v24 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
        v25 = v20;
        v27 = (*(__int64 (__fastcall **)(void *))(*(_QWORD *)v18 + 24LL))(v18);
        if ( (v20[3] & 2) == 0 )
          v25 = (_QWORD *)*v20;
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64 **))(v20[3] & 0xFFFFFFFFFFFFFFF8LL))(
          v25,
          v27,
          v26,
          &v55);
        if ( v55 )
          (*(void (__fastcall **)(__int64 *))(*v55 + 8))(v55);
        v20 += 4;
      }
      while ( v22 != v20 );
      v4 = a1;
    }
  }
  v53 = a3;
  v6 = (unsigned __int8)sub_227BB30(v4 + 32, (__int64 *)&v53, &v54) == 0;
  v28 = v54;
  if ( v6 )
  {
    v39 = *(_DWORD *)(v4 + 48);
    ++*(_QWORD *)(v4 + 32);
    v55 = v28;
    v40 = v39 + 1;
    v41 = *(_DWORD *)(v4 + 56);
    if ( 4 * v40 >= 3 * v41 )
    {
      v41 *= 2;
    }
    else if ( v41 - *(_DWORD *)(v4 + 52) - v40 > v41 >> 3 )
    {
LABEL_47:
      *(_DWORD *)(v4 + 48) = v40;
      if ( *v28 != -4096 )
        --*(_DWORD *)(v4 + 52);
      v42 = v53;
      v28[3] = 0;
      v48 = v28 + 1;
      *v28 = (__int64)v42;
      v28[2] = (__int64)(v28 + 1);
      v28[1] = (__int64)(v28 + 1);
      goto LABEL_27;
    }
    sub_227C6A0(v4 + 32, v41);
    sub_227BB30(v4 + 32, (__int64 *)&v53, &v55);
    v40 = *(_DWORD *)(v4 + 48) + 1;
    v28 = v55;
    goto LABEL_47;
  }
  v48 = v54 + 1;
LABEL_27:
  (*(void (__fastcall **)(__int64 **, void *, __int64 *, __int64, __int64))(*(_QWORD *)v18 + 16LL))(
    &v55,
    v18,
    a3,
    v4,
    a4);
  v29 = (_QWORD *)sub_22077B0(0x20u);
  v29[2] = a2;
  v30 = (__int64)v55;
  v55 = 0;
  v29[3] = v30;
  sub_2208C80(v29, (__int64)v48);
  ++v48[2];
  if ( v55 )
    (*(void (__fastcall **)(__int64 *))(*v55 + 8))(v55);
  if ( v47 )
  {
    v31 = *(_QWORD **)(v47 + 864);
    if ( v31 != &v31[4 * *(unsigned int *)(v47 + 872)] )
    {
      v51 = v4;
      v32 = &v31[4 * *(unsigned int *)(v47 + 872)];
      do
      {
        v55 = 0;
        v33 = (__int64 *)sub_22077B0(0x10u);
        if ( v33 )
        {
          v33[1] = (__int64)a3;
          *v33 = (__int64)&unk_4A08BA8;
        }
        v34 = (__int64)v55;
        v55 = v33;
        if ( v34 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 8LL))(v34);
        v35 = v31;
        v37 = (*(__int64 (__fastcall **)(void *))(*(_QWORD *)v18 + 24LL))(v18);
        if ( (v31[3] & 2) == 0 )
          v35 = (_QWORD *)*v31;
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64 **))(v31[3] & 0xFFFFFFFFFFFFFFF8LL))(
          v35,
          v37,
          v36,
          &v55);
        if ( v55 )
          (*(void (__fastcall **)(__int64 *))(*v55 + 8))(v55);
        v31 += 4;
      }
      while ( v32 != v31 );
      v4 = v51;
    }
  }
  v56 = a3;
  v55 = a2;
  v38 = sub_227BA70(v49, (__int64 *)&v55);
  if ( !v38 )
    v38 = *(_QWORD *)(v4 + 72) + 24LL * *(unsigned int *)(v4 + 88);
  v8 = v48[1];
  *(_QWORD *)(v38 + 16) = v8;
  return *(_QWORD *)(v8 + 24);
}
