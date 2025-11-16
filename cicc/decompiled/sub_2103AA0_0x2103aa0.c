// Function: sub_2103AA0
// Address: 0x2103aa0
//
__int64 __fastcall sub_2103AA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // ebx
  __int64 v7; // rdx
  unsigned int v8; // ecx
  _WORD *v9; // rsi
  __int16 *v10; // rdx
  unsigned __int16 v11; // r14
  __int16 *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // r9d
  __int16 v18; // ax
  unsigned int v19; // r13d
  __int64 v20; // r12
  __int128 v22; // [rsp-18h] [rbp-100h]
  int v23; // [rsp+18h] [rbp-D0h] BYREF
  __int64 v24; // [rsp+20h] [rbp-C8h]
  __int64 v25; // [rsp+28h] [rbp-C0h]
  __int64 v26; // [rsp+30h] [rbp-B8h]
  int *v27; // [rsp+38h] [rbp-B0h]
  unsigned __int64 v28[2]; // [rsp+48h] [rbp-A0h] BYREF
  _BYTE v29[48]; // [rsp+58h] [rbp-90h] BYREF
  _BYTE *v30; // [rsp+88h] [rbp-60h]
  __int64 v31; // [rsp+90h] [rbp-58h]
  _BYTE v32[16]; // [rsp+98h] [rbp-50h] BYREF
  __int64 v33; // [rsp+A8h] [rbp-40h]

  v6 = a4;
  v26 = a3;
  v27 = &v23;
  *((_QWORD *)&v22 + 1) = a3;
  *(_QWORD *)&v22 = a2;
  v28[0] = (unsigned __int64)v29;
  v23 = 0;
  v24 = a2;
  v25 = a2;
  v28[1] = 0x200000000LL;
  v30 = v32;
  v31 = 0x200000000LL;
  v33 = 0;
  sub_1DB8610((__int64)v28, a2, a3, a4, a5, a6, v22, (__int64)&v23);
  v7 = *(_QWORD *)(a1 + 232);
  if ( !v7 )
    BUG();
  v8 = *(_DWORD *)(*(_QWORD *)(v7 + 8) + 24LL * v6 + 16);
  v9 = (_WORD *)(*(_QWORD *)(v7 + 56) + 2LL * (v8 >> 4));
  v10 = v9 + 1;
  v11 = *v9 + v6 * (v8 & 0xF);
LABEL_3:
  v12 = v10;
  if ( v10 )
  {
    while ( 1 )
    {
      v13 = sub_2103840(a1, (__int64)v28, v11);
      if ( (unsigned int)sub_20FD0B0(v13, 1u, v14, v15, v16, v17) )
        break;
      v18 = *v12;
      v10 = 0;
      ++v12;
      if ( !v18 )
        goto LABEL_3;
      v11 += v18;
      if ( !v12 )
        goto LABEL_7;
    }
    v19 = 1;
  }
  else
  {
LABEL_7:
    v19 = 0;
  }
  v20 = v33;
  if ( v33 )
  {
    sub_2102A60(*(_QWORD *)(v33 + 16));
    j_j___libc_free_0(v20, 48);
  }
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  if ( (_BYTE *)v28[0] != v29 )
    _libc_free(v28[0]);
  return v19;
}
