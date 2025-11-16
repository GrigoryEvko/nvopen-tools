// Function: sub_1B8F4A0
// Address: 0x1b8f4a0
//
__int64 __fastcall sub_1B8F4A0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  int v8; // ebx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // edx
  _QWORD *v13; // r8
  __int64 v14; // rax
  _QWORD *v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // r9
  _QWORD *v18; // rax
  int v19; // edx
  unsigned int v20; // r12d
  __int64 v22; // [rsp+0h] [rbp-80h]
  unsigned __int64 v23; // [rsp+8h] [rbp-78h]
  _BYTE *v24; // [rsp+20h] [rbp-60h] BYREF
  __int64 v25; // [rsp+28h] [rbp-58h]
  _BYTE v26[80]; // [rsp+30h] [rbp-50h] BYREF

  sub_14C3B40(a1, a4);
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_8;
  v5 = sub_1648A40(a1);
  v7 = v5 + v6;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v7 >> 4) )
LABEL_20:
      BUG();
LABEL_8:
    v11 = -24;
    goto LABEL_9;
  }
  if ( !(unsigned int)((v7 - sub_1648A40(a1)) >> 4) )
    goto LABEL_8;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_20;
  v8 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v9 = sub_1648A40(a1);
  v11 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v9 + v10 - 4) - v8);
LABEL_9:
  v12 = *(_DWORD *)(a1 + 20);
  v13 = (_QWORD *)(a1 + v11);
  v24 = v26;
  v25 = 0x400000000LL;
  v14 = 24LL * (v12 & 0xFFFFFFF);
  v15 = (_QWORD *)(a1 - v14);
  v16 = v11 + v14;
  v17 = 0xAAAAAAAAAAAAAAABLL * (v16 >> 3);
  if ( (unsigned __int64)v16 > 0x60 )
  {
    v22 = a1 + v11;
    v23 = 0xAAAAAAAAAAAAAAABLL * (v16 >> 3);
    sub_16CD150((__int64)&v24, v26, v23, 8, (int)v13, v17);
    v19 = v25;
    LODWORD(v17) = v23;
    v13 = (_QWORD *)v22;
    v18 = &v24[8 * (unsigned int)v25];
  }
  else
  {
    v18 = v26;
    v19 = 0;
  }
  if ( v13 != v15 )
  {
    do
    {
      if ( v18 )
        *v18 = *v15;
      v15 += 3;
      ++v18;
    }
    while ( v13 != v15 );
    v19 = v25;
  }
  LODWORD(v25) = v17 + v19;
  v20 = sub_14A3590(a3);
  if ( v24 != v26 )
    _libc_free((unsigned __int64)v24);
  return v20;
}
