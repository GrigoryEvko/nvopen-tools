// Function: sub_306A480
// Address: 0x306a480
//
unsigned __int64 __fastcall sub_306A480(__int64 a1, int a2, char a3, __int64 *a4, __int64 a5, int a6, unsigned int a7)
{
  char v10; // al
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // r10
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rbx
  signed __int64 v16; // rcx
  unsigned __int64 result; // rax
  __int64 v18; // rax
  unsigned int v19; // eax
  unsigned __int64 v20; // rdx
  __int64 v21; // r10
  signed __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rcx
  unsigned __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // r12
  signed __int64 v28; // rax
  bool v29; // of
  __int128 v30; // [rsp-18h] [rbp-108h]
  __int64 *v31; // [rsp+0h] [rbp-F0h]
  unsigned __int64 v32; // [rsp+0h] [rbp-F0h]
  __int64 v33; // [rsp+8h] [rbp-E8h]
  __int64 v34; // [rsp+8h] [rbp-E8h]
  __int64 v35; // [rsp+8h] [rbp-E8h]
  __int64 v36; // [rsp+8h] [rbp-E8h]
  __int64 v37; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v38; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v39; // [rsp+8h] [rbp-E8h]
  __int64 v40; // [rsp+18h] [rbp-D8h] BYREF
  unsigned __int64 v41; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-C8h]
  char *v43; // [rsp+38h] [rbp-B8h]
  char v44; // [rsp+48h] [rbp-A8h] BYREF
  char *v45; // [rsp+68h] [rbp-88h]
  char v46; // [rsp+78h] [rbp-78h] BYREF

  v10 = *(_BYTE *)(a5 + 8);
  if ( v10 != 17 || a2 != 13 || !a3 )
  {
LABEL_4:
    v11 = *(_DWORD *)(a5 + 32);
    BYTE4(v40) = v10 == 18;
    LODWORD(v40) = v11;
    v12 = sub_BCE1B0(a4, v40);
    v13 = v12;
    if ( (a6 & 1) != 0 )
    {
      v33 = v12;
      v14 = sub_3069790(a1, a2, (_QWORD **)v12, a7);
      v13 = v33;
      v15 = v14;
      goto LABEL_6;
    }
    if ( *(_BYTE *)(v12 + 8) == 18 )
    {
      v15 = 0;
      goto LABEL_6;
    }
    v19 = *(_DWORD *)(v12 + 32);
    v42 = v19;
    if ( v19 > 0x40 )
    {
      v37 = v13;
      sub_C43690((__int64)&v41, -1, 1);
      v13 = v37;
    }
    else
    {
      v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
      if ( !v19 )
        v20 = 0;
      v41 = v20;
    }
    v35 = v13;
    v32 = sub_3064F80(a1, v13, (__int64 *)&v41, 0, 1);
    v21 = v35;
    if ( v42 > 0x40 && v41 )
    {
      j_j___libc_free_0_0(v41);
      v21 = v35;
    }
    v36 = v21;
    v22 = sub_3075ED0(a1, a2, *(_QWORD *)(v21 + 24), a7, 0, 0, 0, 0, 0);
    v13 = v36;
    v24 = *(unsigned int *)(v36 + 32) * v22;
    if ( is_mul_ok(*(unsigned int *)(v36 + 32), v22) )
    {
      v25 = v24 + v32;
      if ( __OFADD__(v24, v32) )
      {
        v15 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v24 > 0 )
          goto LABEL_6;
        goto LABEL_22;
      }
      goto LABEL_28;
    }
    if ( *(_DWORD *)(v36 + 32) && v22 > 0 )
    {
      if ( v23 == 1 )
      {
        v15 = v32 + 0x7FFFFFFFFFFFFFFFLL;
        if ( __OFADD__(0x7FFFFFFFFFFFFFFFLL, v32) )
          v15 = 0x7FFFFFFFFFFFFFFFLL;
        goto LABEL_6;
      }
      v25 = v32 + 0x7FFFFFFFFFFFFFFFLL;
      v15 = 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v32) )
LABEL_28:
        v15 = v25;
LABEL_6:
      v16 = sub_3065900(a1, (unsigned int)(a3 == 0) + 39, v13, a5, 0, a7, 0);
      result = v16 + v15;
      if ( __OFADD__(v16, v15) )
      {
        result = 0x7FFFFFFFFFFFFFFFLL;
        if ( v16 <= 0 )
          return 0x8000000000000000LL;
      }
      return result;
    }
    if ( v23 == 1 )
    {
      v15 = v32 + 0x8000000000000000LL;
      if ( !__OFADD__(v32, 0x8000000000000000LL) )
        goto LABEL_6;
    }
    else
    {
      v15 = v32 + 0x8000000000000000LL;
      if ( !__OFADD__(v32, 0x8000000000000000LL) )
        goto LABEL_6;
    }
LABEL_22:
    v15 = 0x8000000000000000LL;
    goto LABEL_6;
  }
  v31 = a4;
  v34 = *(_QWORD *)(a5 + 24);
  v18 = sub_BCB2A0(*(_QWORD **)a5);
  a4 = v31;
  if ( v34 != v18 )
  {
    v10 = *(_BYTE *)(a5 + 8);
    goto LABEL_4;
  }
  v26 = sub_BCCE00((_QWORD *)*v31, *(_DWORD *)(a5 + 32));
  v40 = v26;
  *((_QWORD *)&v30 + 1) = 1;
  *(_QWORD *)&v30 = 0;
  sub_DF8CB0((__int64)&v41, 66, v26, (char *)&v40, 1, a6, 0, v30);
  v27 = sub_3078340(a1, &v41, a7);
  v28 = sub_3065900(a1, 0x31u, v26, a5, 0, a7, 0);
  v29 = __OFADD__(v27, v28);
  result = v27 + v28;
  if ( v29 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v27 <= 0 )
      result = 0x8000000000000000LL;
  }
  if ( v45 != &v46 )
  {
    v38 = result;
    _libc_free((unsigned __int64)v45);
    result = v38;
  }
  if ( v43 != &v44 )
  {
    v39 = result;
    _libc_free((unsigned __int64)v43);
    return v39;
  }
  return result;
}
