// Function: sub_18B6A10
// Address: 0x18b6a10
//
__int64 __fastcall sub_18B6A10(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        char *a7,
        size_t a8,
        unsigned int a9)
{
  _BYTE *v12; // r14
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // r12
  __int64 v20; // rsi
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rax
  int v29; // [rsp+28h] [rbp-98h]
  __int64 v30; // [rsp+30h] [rbp-90h] BYREF
  __int16 v31; // [rsp+40h] [rbp-80h]
  __int64 v32[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v33[2]; // [rsp+60h] [rbp-60h] BYREF
  int v34; // [rsp+70h] [rbp-50h]
  int v35; // [rsp+84h] [rbp-3Ch]

  v31 = 260;
  v30 = *a1 + 240;
  sub_16E1010((__int64)v32, (__int64)&v30);
  if ( (unsigned int)(v34 - 31) <= 1 && v35 == 2 )
  {
    if ( (_QWORD *)v32[0] != v33 )
      j_j___libc_free_0(v32[0], v33[0] + 1LL);
    v12 = sub_18B6670(*a1, a1[5], a2, a3, a4, a5, a7, a8);
    v13 = sub_1649C60((__int64)v12);
    v14 = sub_15A4180((unsigned __int64)v12, (__int64 **)a6, 0);
    if ( !sub_1626AA0(v13, 21) )
    {
      v15 = a1[9];
      if ( *(_DWORD *)(v15 + 8) >> 8 == *(_DWORD *)(a6 + 8) >> 8 )
      {
        v24 = sub_159C470(v15, -1, 0);
        v25 = sub_1624210(v24);
        v18 = a1[9];
        v20 = -1;
        v19 = (__int64)v25;
      }
      else
      {
        v29 = *(_DWORD *)(a6 + 8) >> 8;
        v16 = sub_159C470(v15, 0, 0);
        v17 = sub_1624210(v16);
        v18 = a1[9];
        v19 = (__int64)v17;
        v20 = 1LL << v29;
      }
      v21 = sub_159C470(v18, v20, 0);
      v22 = sub_1624210(v21);
      v32[0] = v19;
      v32[1] = (__int64)v22;
      v23 = sub_1627350(*(__int64 **)*a1, v32, (__int64 *)2, 0, 1);
      sub_16270B0(v13, 0x15u, v23);
    }
    return v14;
  }
  else
  {
    if ( (_QWORD *)v32[0] != v33 )
      j_j___libc_free_0(v32[0], v33[0] + 1LL);
    return sub_159C470(a6, a9, 0);
  }
}
