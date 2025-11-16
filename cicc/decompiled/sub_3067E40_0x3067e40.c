// Function: sub_3067E40
// Address: 0x3067e40
//
unsigned __int64 __fastcall sub_3067E40(
        __int64 a1,
        int a2,
        _QWORD **a3,
        unsigned __int8 a4,
        char a5,
        char a6,
        int a7,
        unsigned int a8)
{
  unsigned int v9; // r15d
  unsigned __int8 v11; // r11
  __int64 v12; // rcx
  signed __int64 v13; // rax
  __int64 v14; // r10
  __int64 v15; // r13
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  unsigned __int64 v19; // rax
  bool v20; // of
  unsigned __int64 v21; // rax
  unsigned __int64 result; // rax
  unsigned int v23; // eax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rax
  unsigned int v28; // edx
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rax
  unsigned int v31; // eax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // r12
  __int64 v34; // r12
  __int64 v35; // [rsp+10h] [rbp-70h]
  unsigned __int64 v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+18h] [rbp-68h]
  unsigned __int64 v38; // [rsp+28h] [rbp-58h]
  unsigned __int64 v39; // [rsp+28h] [rbp-58h]
  __int64 v40; // [rsp+28h] [rbp-58h]
  __int64 v42; // [rsp+38h] [rbp-48h]
  unsigned __int8 v44; // [rsp+38h] [rbp-48h]
  unsigned __int64 v45; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v46; // [rsp+48h] [rbp-38h]

  if ( *((_BYTE *)a3 + 8) == 18 )
    return 0;
  v38 = 0;
  v9 = *((_DWORD *)a3 + 8);
  v11 = a4;
  if ( a6 )
  {
    v26 = (__int64 *)sub_BCE3C0(*a3, 0);
    v27 = sub_BCDA70(v26, v9);
    v11 = a4;
    if ( *(_BYTE *)(v27 + 8) != 18 )
    {
      v28 = *(_DWORD *)(v27 + 32);
      v46 = v28;
      if ( v28 > 0x40 )
      {
        v40 = v27;
        sub_C43690((__int64)&v45, -1, 1);
        v11 = a4;
        v27 = v40;
      }
      else
      {
        v29 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v28;
        if ( !v28 )
          v29 = 0;
        v45 = v29;
      }
      v44 = v11;
      v30 = sub_3064F80(a1, v27, (__int64 *)&v45, 0, 1);
      v11 = v44;
      if ( v46 > 0x40 && v45 )
      {
        v39 = v30;
        j_j___libc_free_0_0(v45);
        v30 = v39;
        v11 = v44;
      }
      v38 = v30;
    }
  }
  v12 = v11;
  BYTE1(v12) = 1;
  v13 = sub_30670E0(a1, a2, (__int64)a3[3], v12, a8, a7);
  if ( is_mul_ok(v13, v9) )
  {
    v42 = v13 * v9;
  }
  else if ( !v9 || (v42 = 0x7FFFFFFFFFFFFFFFLL, v13 <= 0) )
  {
    v42 = 0x8000000000000000LL;
  }
  if ( *((_BYTE *)a3 + 8) == 18 )
  {
    v14 = 0;
  }
  else
  {
    v23 = *((_DWORD *)a3 + 8);
    v46 = v23;
    if ( v23 > 0x40 )
    {
      sub_C43690((__int64)&v45, -1, 1);
    }
    else
    {
      v24 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v23;
      if ( !v23 )
        v24 = 0;
      v45 = v24;
    }
    v25 = sub_3064F80(a1, (__int64)a3, (__int64 *)&v45, a2 != 33, a2 == 33);
    v14 = v25;
    if ( v46 > 0x40 && v45 )
    {
      v36 = v25;
      j_j___libc_free_0_0(v45);
      v14 = v36;
    }
  }
  if ( a5 )
  {
    v35 = v14;
    v15 = v9 * ((a7 == 0) + 1LL);
    v16 = (__int64 *)sub_BCB2A0(*a3);
    v17 = sub_BCDA70(v16, v9);
    v14 = v35;
    v18 = v17;
    if ( *(_BYTE *)(v17 + 8) != 18 )
    {
      v31 = *(_DWORD *)(v17 + 32);
      v46 = v31;
      if ( v31 > 0x40 )
      {
        sub_C43690((__int64)&v45, -1, 1);
        v14 = v35;
      }
      else
      {
        v32 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v31;
        if ( !v31 )
          v32 = 0;
        v45 = v32;
      }
      v37 = v14;
      v33 = sub_3064F80(a1, v18, (__int64 *)&v45, 0, 1);
      v14 = v37;
      if ( v46 > 0x40 && v45 )
      {
        j_j___libc_free_0_0(v45);
        v14 = v37;
      }
      v20 = __OFADD__(v15, v33);
      v34 = v15 + v33;
      if ( v20 )
      {
        if ( v15 )
          v15 = 0x7FFFFFFFFFFFFFFFLL;
        else
          v15 = 0x8000000000000000LL;
      }
      else
      {
        v15 = v34;
      }
    }
  }
  else
  {
    v15 = 0;
  }
  v19 = v42 + v38;
  if ( __OFADD__(v42, v38) )
  {
    v19 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v42 <= 0 )
      v19 = 0x8000000000000000LL;
  }
  v20 = __OFADD__(v14, v19);
  v21 = v14 + v19;
  if ( v20 )
  {
    v21 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v14 <= 0 )
      v21 = 0x8000000000000000LL;
  }
  v20 = __OFADD__(v15, v21);
  result = v15 + v21;
  if ( v20 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v15 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
