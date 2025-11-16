// Function: sub_30ED730
// Address: 0x30ed730
//
void __fastcall sub_30ED730(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        unsigned __int64 *a9)
{
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  char *v13; // r14
  unsigned __int64 *v14; // r13
  unsigned __int64 *v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rax
  _BYTE *v18; // rdi
  __int8 *v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  _BYTE *v23; // rsi
  _BYTE *v24; // r9
  unsigned __int64 v25; // rax
  _BYTE *v26; // r15
  _BYTE *v27; // rdi
  size_t v28; // r14
  __int8 *v29; // rax
  __int8 *v30; // rdi
  unsigned __int64 v31[3]; // [rsp+8h] [rbp-228h] BYREF
  char v32; // [rsp+24h] [rbp-20Ch] BYREF
  _BYTE v33[11]; // [rsp+25h] [rbp-20Bh] BYREF
  __int8 *v34; // [rsp+30h] [rbp-200h] BYREF
  size_t v35; // [rsp+38h] [rbp-1F8h]
  _QWORD v36[2]; // [rsp+40h] [rbp-1F0h] BYREF
  _QWORD v37[10]; // [rsp+50h] [rbp-1E0h] BYREF
  unsigned __int64 *v38; // [rsp+A0h] [rbp-190h]
  unsigned int v39; // [rsp+A8h] [rbp-188h]
  char v40; // [rsp+B0h] [rbp-180h] BYREF

  v10 = *a1;
  v11 = sub_B2BE50(*a1);
  if ( !sub_B6EA50(v11) )
  {
    v20 = sub_B2BE50(v10);
    v21 = sub_B6F970(v20);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v21 + 48LL))(v21) )
      return;
  }
  sub_B17560((__int64)v37, (__int64)"kernel-info", *a7, a7[1], a8);
  sub_B18290((__int64)v37, "in ", 3u);
  sub_30ED1B0((__int64)v37, *(_QWORD *)(a8 + 40), (unsigned __int8 *)a8, "function", 8u);
  sub_B18290((__int64)v37, ", ", 2u);
  sub_B18290((__int64)v37, (__int8 *)*a7, a7[1]);
  sub_B18290((__int64)v37, " = ", 3u);
  v12 = *a9;
  if ( (*a9 & 0x8000000000000000LL) != 0LL )
  {
    v22 = -(__int64)v12;
    v23 = v33;
    do
    {
      v24 = v23--;
      *v23 = v22 % 0xA + 48;
      v25 = v22;
      v22 /= 0xAu;
    }
    while ( v25 > 9 );
    v26 = v24 - 2;
    *(v23 - 1) = 45;
    v27 = (_BYTE *)(v33 - (v24 - 2));
    v34 = (__int8 *)v36;
    v31[0] = (unsigned __int64)v27;
    v28 = (size_t)v27;
    if ( (unsigned __int64)v27 > 0xF )
    {
      v34 = (__int8 *)sub_22409D0((__int64)&v34, v31, 0);
      v30 = v34;
      v36[0] = v31[0];
    }
    else
    {
      if ( v27 == (_BYTE *)1 )
      {
        LOBYTE(v36[0]) = 45;
        v29 = (__int8 *)v36;
LABEL_31:
        v35 = v28;
        v29[v28] = 0;
        goto LABEL_7;
      }
      if ( !v27 )
      {
        v29 = (__int8 *)v36;
        goto LABEL_31;
      }
      v30 = (__int8 *)v36;
    }
    memcpy(v30, v26, v28);
    v28 = v31[0];
    v29 = v34;
    goto LABEL_31;
  }
  if ( *a9 )
  {
    v13 = v33;
    do
    {
      *--v13 = v12 % 0xA + 48;
      v17 = v12;
      v12 /= 0xAu;
    }
    while ( v17 > 9 );
    v18 = (_BYTE *)(v33 - v13);
    v34 = (__int8 *)v36;
    v31[0] = v33 - v13;
    if ( (unsigned __int64)(v33 - v13) <= 0xF )
    {
      if ( v18 == (_BYTE *)1 )
        goto LABEL_5;
      if ( !v18 )
        goto LABEL_6;
      v19 = (__int8 *)v36;
    }
    else
    {
      v34 = (__int8 *)sub_22409D0((__int64)&v34, v31, 0);
      v19 = v34;
      v36[0] = v31[0];
    }
    memcpy(v19, v13, v33 - v13);
    goto LABEL_6;
  }
  v32 = 48;
  v13 = &v32;
  v34 = (__int8 *)v36;
  v31[0] = 1;
LABEL_5:
  LOBYTE(v36[0]) = *v13;
LABEL_6:
  v35 = v31[0];
  v34[v31[0]] = 0;
LABEL_7:
  sub_B18290((__int64)v37, v34, v35);
  if ( v34 != (__int8 *)v36 )
    j_j___libc_free_0((unsigned __int64)v34);
  sub_1049740(a1, (__int64)v37);
  v14 = v38;
  v37[0] = &unk_49D9D40;
  v15 = &v38[10 * v39];
  if ( v38 != v15 )
  {
    do
    {
      v15 -= 10;
      v16 = v15[4];
      if ( (unsigned __int64 *)v16 != v15 + 6 )
        j_j___libc_free_0(v16);
      if ( (unsigned __int64 *)*v15 != v15 + 2 )
        j_j___libc_free_0(*v15);
    }
    while ( v14 != v15 );
    v15 = v38;
  }
  if ( v15 != (unsigned __int64 *)&v40 )
    _libc_free((unsigned __int64)v15);
}
