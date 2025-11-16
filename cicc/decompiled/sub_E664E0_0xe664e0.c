// Function: sub_E664E0
// Address: 0xe664e0
//
__int64 __fastcall sub_E664E0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 **v5; // rcx
  char v6; // r9
  bool v7; // zf
  __int64 v8; // rdx
  _QWORD *v9; // rsi
  __int64 result; // rax
  _BYTE *v11; // rbx
  _BYTE *v12; // r12
  _BYTE *v13; // rdi
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  __int64 *v16; // rbx
  __int64 *v17; // r12
  __int64 *v18; // rdi
  char v19; // [rsp+Fh] [rbp-201h]
  __int64 **v20; // [rsp+10h] [rbp-200h]
  char v21; // [rsp+27h] [rbp-1E9h] BYREF
  __int64 **v22; // [rsp+28h] [rbp-1E8h] BYREF
  __int64 *v23; // [rsp+30h] [rbp-1E0h] BYREF
  __int64 *v24; // [rsp+38h] [rbp-1D8h]
  __int64 v25; // [rsp+40h] [rbp-1D0h]
  _QWORD *v26; // [rsp+48h] [rbp-1C8h]
  _QWORD *v27; // [rsp+50h] [rbp-1C0h]
  __int64 v28; // [rsp+58h] [rbp-1B8h]
  __int64 v29; // [rsp+60h] [rbp-1B0h]
  __int64 v30; // [rsp+68h] [rbp-1A8h]
  _QWORD v31[2]; // [rsp+70h] [rbp-1A0h] BYREF
  _QWORD *v32; // [rsp+80h] [rbp-190h]
  __int64 v33; // [rsp+88h] [rbp-188h]
  _QWORD v34[3]; // [rsp+90h] [rbp-180h] BYREF
  int v35; // [rsp+A8h] [rbp-168h]
  _QWORD *v36; // [rsp+B0h] [rbp-160h]
  __int64 v37; // [rsp+B8h] [rbp-158h]
  _QWORD v38[2]; // [rsp+C0h] [rbp-150h] BYREF
  _QWORD *v39; // [rsp+D0h] [rbp-140h]
  __int64 v40; // [rsp+D8h] [rbp-138h]
  _QWORD v41[2]; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v42; // [rsp+F0h] [rbp-120h]
  __int64 v43; // [rsp+F8h] [rbp-118h]
  __int64 v44; // [rsp+100h] [rbp-110h]
  _BYTE *v45; // [rsp+108h] [rbp-108h]
  __int64 v46; // [rsp+110h] [rbp-100h]
  _BYTE v47[248]; // [rsp+118h] [rbp-F8h] BYREF

  v4 = a1;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  if ( a2 )
  {
    v5 = *(__int64 ***)(a1 + 80);
    v6 = 0;
    if ( !v5 )
    {
      v5 = *(__int64 ***)(a1 + 88);
      if ( !v5 )
        BUG();
      v6 = 1;
    }
  }
  else
  {
    v6 = 0;
    v5 = &v23;
  }
  v7 = *(_QWORD *)(a3 + 16) == 0;
  v31[0] = 0;
  v32 = v34;
  v8 = 0x400000000LL;
  v31[1] = 0;
  v33 = 0;
  LOBYTE(v34[0]) = 0;
  v34[2] = 0;
  v35 = 0;
  v36 = v38;
  v37 = 0;
  LOBYTE(v38[0]) = 0;
  v39 = v41;
  v40 = 0;
  LOBYTE(v41[0]) = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = v47;
  v46 = 0x400000000LL;
  v22 = v5;
  if ( v7
    || (v19 = v6,
        a2 = v31,
        a1 = a3,
        v20 = v5,
        (*(void (__fastcall **)(__int64, _QWORD *, __int64 ***))(a3 + 24))(a3, v31, &v22),
        v7 = *(_QWORD *)(v4 + 136) == 0,
        v21 = v19,
        v7) )
  {
    sub_4263D6(a1, a2, v8);
  }
  v9 = v31;
  (*(void (__fastcall **)(__int64, _QWORD *, char *, __int64 **, __int64))(v4 + 144))(v4 + 120, v31, &v21, v20, v4 + 96);
  result = (unsigned int)v46;
  v11 = v45;
  v12 = &v45[48 * (unsigned int)v46];
  if ( v45 != v12 )
  {
    do
    {
      v12 -= 48;
      v13 = (_BYTE *)*((_QWORD *)v12 + 2);
      result = (__int64)(v12 + 32);
      if ( v13 != v12 + 32 )
      {
        v9 = (_QWORD *)(*((_QWORD *)v12 + 4) + 1LL);
        result = j_j___libc_free_0(v13, v9);
      }
    }
    while ( v11 != v12 );
    v12 = v45;
  }
  if ( v12 != v47 )
    result = _libc_free(v12, v9);
  if ( v42 )
    result = j_j___libc_free_0(v42, v44 - v42);
  if ( v39 != v41 )
    result = j_j___libc_free_0(v39, v41[0] + 1LL);
  if ( v36 != v38 )
    result = j_j___libc_free_0(v36, v38[0] + 1LL);
  if ( v32 != v34 )
    result = j_j___libc_free_0(v32, v34[0] + 1LL);
  v14 = v27;
  v15 = v26;
  if ( v27 != v26 )
  {
    do
    {
      result = (__int64)(v15 + 2);
      if ( (_QWORD *)*v15 != v15 + 2 )
        result = j_j___libc_free_0(*v15, v15[2] + 1LL);
      v15 += 4;
    }
    while ( v14 != v15 );
    v15 = v26;
  }
  if ( v15 )
    result = j_j___libc_free_0(v15, v28 - (_QWORD)v15);
  v16 = v24;
  v17 = v23;
  if ( v24 != v23 )
  {
    do
    {
      v18 = v17;
      v17 += 3;
      result = sub_C8EE20(v18);
    }
    while ( v16 != v17 );
    v17 = v23;
  }
  if ( v17 )
    return j_j___libc_free_0(v17, v25 - (_QWORD)v17);
  return result;
}
