// Function: sub_255B2D0
// Address: 0x255b2d0
//
__int64 __fastcall sub_255B2D0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // rdi
  __int64 *v7; // r15
  int v8; // eax
  unsigned int v9; // r15d
  unsigned int v11; // esi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int v17; // r15d
  unsigned __int64 v18; // rdi
  __int64 *v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r15
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v26; // [rsp+10h] [rbp-50h] BYREF
  __int64 v27; // [rsp+18h] [rbp-48h]
  _BYTE v28[64]; // [rsp+20h] [rbp-40h] BYREF

  v2 = (__int64 *)(a1 + 72);
  v27 = 0x100000000LL;
  v5 = *(_QWORD *)(a1 + 72);
  v26 = (__int64 *)v28;
  v6 = v5 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (v5 & 3) == 3 )
    v6 = *(_QWORD *)(v6 + 24);
  v7 = (__int64 *)sub_BD5C60(v6);
  if ( (unsigned __int8)sub_2509800(v2) != 4 )
    goto LABEL_4;
  if ( (unsigned __int8)*(_DWORD *)(a1 + 100) != 255 && (*(_DWORD *)(a1 + 100) & 0xFC) != 0xFC )
  {
    if ( (*(_DWORD *)(a1 + 100) & 0xDC) == 0xDC )
    {
      v11 = 12;
    }
    else
    {
      v11 = 3;
      if ( (*(_DWORD *)(a1 + 100) & 0xEC) != 0xEC )
      {
        if ( (*(_DWORD *)(a1 + 100) & 0xCC) != 0xCC )
        {
LABEL_4:
          v8 = v27;
          goto LABEL_5;
        }
        sub_255AA10(&v25, 0);
        v11 = v25 | 0xF;
      }
    }
    v12 = sub_A77AB0(v7, v11);
    sub_255A480((__int64)&v26, v12, v13, v14, v15, v16);
    goto LABEL_4;
  }
  sub_255AA10(&v25, 0);
  v22 = sub_A77AB0(v7, v25);
  v23 = (unsigned int)v27;
  v24 = (unsigned int)v27 + 1LL;
  if ( v24 > HIDWORD(v27) )
  {
    sub_C8D5F0((__int64)&v26, v28, v24, 8u, v20, v21);
    v23 = (unsigned int)v27;
  }
  v26[v23] = v22;
  v8 = v27 + 1;
  LODWORD(v27) = v27 + 1;
LABEL_5:
  v9 = 1;
  if ( v8 == 1 )
  {
    v17 = sub_A71E40(v26);
    v18 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
      v18 = *(_QWORD *)(v18 + 24);
    v19 = (__int64 *)sub_BD5C60(v18);
    v25 = sub_A77AB0(v19, v17);
    v9 = sub_2516380(a2, v2, (__int64)&v25, 1, 0);
  }
  if ( v26 != (__int64 *)v28 )
    _libc_free((unsigned __int64)v26);
  return v9;
}
