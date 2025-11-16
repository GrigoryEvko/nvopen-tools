// Function: sub_11DF4F0
// Address: 0x11df4f0
//
__int64 __fastcall sub_11DF4F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // r10
  unsigned __int64 v7; // rax
  __int64 v8; // r15
  unsigned int v9; // eax
  __int64 v10; // rax
  _QWORD **v11; // rbx
  unsigned int v12; // eax
  __int64 v13; // rax
  _BYTE *v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rax
  _BYTE *v17; // r15
  unsigned __int8 *v18; // rax
  _QWORD *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // [rsp+8h] [rbp-88h]
  __int64 v24; // [rsp+8h] [rbp-88h]
  _QWORD **v25; // [rsp+10h] [rbp-80h]
  __int64 *v26; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  __int64 v28; // [rsp+18h] [rbp-78h]
  _BYTE *v29; // [rsp+28h] [rbp-68h] BYREF
  unsigned int v30[8]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v31; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v5 = 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = *(_QWORD *)(a2 + v5);
  if ( *(_QWORD *)(a2 + 16) )
  {
    if ( v6 != v4 )
    {
      v27 = *(_QWORD *)(a2 + v5);
      v7 = sub_98B430(v6, 8u);
      v8 = v7;
      if ( v7 )
      {
        v30[0] = 1;
        sub_11DA2E0(a2, v30, 1, v7);
        v23 = *(__int64 **)(a1 + 24);
        v25 = (_QWORD **)sub_B43CA0(a2);
        v9 = sub_97FA80(*v23, (__int64)v25);
        v10 = sub_BCCE00(*v25, v9);
        v24 = sub_ACD640(v10, v8, 0);
        v26 = *(__int64 **)(a1 + 24);
        v31 = 257;
        v11 = (_QWORD **)sub_B43CA0(a2);
        v12 = sub_97FA80(*v26, (__int64)v11);
        v13 = sub_BCCE00(*v11, v12);
        v14 = (_BYTE *)sub_ACD640(v13, v8 - 1, 0);
        v15 = *(_QWORD **)(a3 + 72);
        v29 = v14;
        v16 = sub_BCB2B0(v15);
        v17 = (_BYTE *)sub_921130((unsigned int **)a3, v16, v4, &v29, 1, (__int64)v30, 3u);
        v18 = (unsigned __int8 *)sub_B343C0(a3, 0xEEu, v4, 0x100u, v27, 0x100u, v24, 0, 0, 0, 0, 0);
        sub_11DAF00(v18, a2);
        return (__int64)v17;
      }
      return 0;
    }
    v28 = *(_QWORD *)(a2 + v5);
    v29 = (_BYTE *)sub_11CA050(v6, a3, *(_QWORD *)(a1 + 16), *(__int64 **)(a1 + 24));
    v17 = v29;
    if ( v29 )
    {
      v20 = *(_QWORD **)(a3 + 72);
      v31 = 257;
      v21 = sub_BCB2B0(v20);
      return sub_921130((unsigned int **)a3, v21, v28, &v29, 1, (__int64)v30, 3u);
    }
  }
  else
  {
    v22 = sub_11CA290(v4, *(_QWORD *)(a2 + v5), a3, *(__int64 **)(a1 + 24));
    v17 = (_BYTE *)v22;
    if ( !v22 )
      return 0;
    if ( *(_BYTE *)v22 == 85 )
      *(_WORD *)(v22 + 2) = *(_WORD *)(v22 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  }
  return (__int64)v17;
}
