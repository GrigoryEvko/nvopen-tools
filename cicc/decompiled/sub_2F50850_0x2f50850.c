// Function: sub_2F50850
// Address: 0x2f50850
//
__int64 __fastcall sub_2F50850(_QWORD *a1, __int64 a2, int a3)
{
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r11d
  int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned int v10; // r13d
  __int16 *v11; // r12
  __int64 v12; // rcx
  __int64 v13; // rax
  int v14; // edx
  __int64 v16; // r12
  int *v17; // r10
  unsigned __int16 *v18; // rsi
  int v19; // [rsp+4h] [rbp-14Ch]
  int v22; // [rsp+1Ch] [rbp-134h]
  unsigned __int16 *v23; // [rsp+20h] [rbp-130h] BYREF
  __int64 v24; // [rsp+28h] [rbp-128h]
  _BYTE v25[32]; // [rsp+38h] [rbp-118h] BYREF
  __int64 v26; // [rsp+58h] [rbp-F8h]
  int v27; // [rsp+68h] [rbp-E8h]
  _QWORD v28[4]; // [rsp+70h] [rbp-E0h] BYREF
  _BYTE *v29; // [rsp+90h] [rbp-C0h]
  __int64 v30; // [rsp+98h] [rbp-B8h]
  _BYTE v31[64]; // [rsp+A0h] [rbp-B0h] BYREF
  _BYTE *v32; // [rsp+E0h] [rbp-70h]
  __int64 v33; // [rsp+E8h] [rbp-68h]
  _BYTE v34[32]; // [rsp+F0h] [rbp-60h] BYREF
  __int16 v35; // [rsp+110h] [rbp-40h]
  __int64 v36; // [rsp+114h] [rbp-3Ch]

  sub_34B8230(&v23, *(unsigned int *)(a2 + 112), a1[5], a1[8], a1[3]);
  v5 = v27;
  v6 = -(int)v24;
  v19 = v27;
  if ( v27 == -(int)v24 )
  {
LABEL_23:
    if ( v23 != (unsigned __int16 *)v25 )
      _libc_free((unsigned __int64)v23);
    return 0;
  }
  else
  {
    while ( 1 )
    {
      v7 = v6 < 0 ? v23[v24 + v6] : *(unsigned __int16 *)(v26 + 2LL * v6);
      if ( (_DWORD)v7 != a3 )
        break;
LABEL_17:
      if ( v5 > v6 && v5 > ++v6 && v6 >= 0 )
      {
        v16 = v26;
        v4 = v6;
        v17 = (int *)v28;
        do
        {
          v6 = v4;
          if ( (unsigned int)*(unsigned __int16 *)(v16 + 2 * v4) - 1 > 0x3FFFFFFE )
            break;
          LODWORD(v28[0]) = *(unsigned __int16 *)(v16 + 2 * v4);
          v18 = &v23[v24];
          if ( v18 == sub_2F4C810(v23, (__int64)v18, v17) )
            break;
          ++v4;
          ++v6;
        }
        while ( v5 > (int)v4 );
      }
      if ( v19 == v6 )
        goto LABEL_23;
    }
    v8 = a1[7];
    v9 = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 24 * v7 + 16);
    v10 = v9 & 0xFFF;
    v11 = (__int16 *)(*(_QWORD *)(v8 + 56) + 2LL * (v9 >> 12));
    do
    {
      if ( !v11 )
        break;
      v12 = a1[3];
      v13 = *(_QWORD *)(v12 + 48);
      v29 = v31;
      v28[3] = 0;
      v36 = 0;
      v28[0] = v13 + 216LL * v10;
      v32 = v34;
      v28[1] = a2;
      v35 = 0;
      v30 = 0x400000000LL;
      v33 = 0x400000000LL;
      v22 = sub_2E1AC90((__int64)v28, 1u, 0, v12, v3, v4);
      if ( v32 != v34 )
        _libc_free((unsigned __int64)v32);
      if ( v29 != v31 )
        _libc_free((unsigned __int64)v29);
      if ( v22 )
      {
        v5 = v27;
        goto LABEL_17;
      }
      v14 = *v11++;
      v10 += v14;
    }
    while ( (_WORD)v14 );
    if ( v23 != (unsigned __int16 *)v25 )
      _libc_free((unsigned __int64)v23);
    return 1;
  }
}
