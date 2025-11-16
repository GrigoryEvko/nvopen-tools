// Function: sub_278B200
// Address: 0x278b200
//
__int64 __fastcall sub_278B200(_BYTE **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  __int64 v5; // r8
  int v6; // eax
  _QWORD *v7; // r12
  __int64 v8; // rax
  __int64 v10; // rax
  _BYTE *v11; // r14
  _BYTE *v12; // rbx
  __int64 v13; // r15
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  const char *v21; // rsi
  const char **v22; // r13
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  __int64 v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+8h] [rbp-68h]
  __int64 v31; // [rsp+8h] [rbp-68h]
  const char *v32[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v33; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 8);
  v5 = sub_B43CC0(a2);
  v6 = *((_DWORD *)a1 + 2);
  if ( !v6 )
  {
    v7 = *a1;
    if ( *((_QWORD *)*a1 + 1) != v4 )
    {
      v8 = sub_B43CB0(a2);
      return sub_2A89D00(v7, *((unsigned int *)a1 + 3), v4, a3, v8);
    }
    return (__int64)v7;
  }
  if ( v6 == 1 )
  {
    if ( v4 != *((_QWORD *)*a1 + 1) || *((_DWORD *)a1 + 3) )
    {
      v30 = (__int64)*a1;
      v23 = sub_B43CB0(a2);
      v24 = sub_2A89D00(v30, *((unsigned int *)a1 + 3), v4, a3, v23);
      v25 = v30;
      v7 = (_QWORD *)v24;
      if ( (*(_BYTE *)(v30 + 7) & 0x20) == 0 || (v26 = sub_B91C10(v30, 29), v25 = v30, !v26) )
      {
        v32[0] = (const char *)0xD0000000CLL;
        v32[1] = (const char *)0x1000000006LL;
        sub_B9ADA0(v25, (unsigned int *)v32, 4);
      }
    }
    else
    {
      v31 = (__int64)*a1;
      sub_F57030(*a1, a2, 0);
      return v31;
    }
    return (__int64)v7;
  }
  if ( v6 != 2 )
  {
    if ( v6 != 4 )
      BUG();
    v10 = (__int64)*a1;
    v33 = 257;
    v11 = a1[3];
    v12 = a1[2];
    v13 = *(_QWORD *)(v10 - 96);
    v29 = v10 + 24;
    v14 = sub_BD2C40(72, 3u);
    v7 = v14;
    if ( v14 )
    {
      sub_B44260((__int64)v14, *((_QWORD *)v12 + 1), 57, 3u, v29, 0);
      if ( *(v7 - 12) )
      {
        v15 = *(v7 - 11);
        *(_QWORD *)*(v7 - 10) = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = *(v7 - 10);
      }
      *(v7 - 12) = v13;
      if ( v13 )
      {
        v16 = *(_QWORD *)(v13 + 16);
        *(v7 - 11) = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = v7 - 11;
        *(v7 - 10) = v13 + 16;
        *(_QWORD *)(v13 + 16) = v7 - 12;
      }
      if ( *(v7 - 8) )
      {
        v17 = *(v7 - 7);
        *(_QWORD *)*(v7 - 6) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = *(v7 - 6);
      }
      *(v7 - 8) = v12;
      v18 = *((_QWORD *)v12 + 2);
      *(v7 - 7) = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = v7 - 7;
      *(v7 - 6) = v12 + 16;
      *((_QWORD *)v12 + 2) = v7 - 8;
      if ( *(v7 - 4) )
      {
        v19 = *(v7 - 3);
        *(_QWORD *)*(v7 - 2) = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = *(v7 - 2);
      }
      *(v7 - 4) = v11;
      if ( v11 )
      {
        v20 = *((_QWORD *)v11 + 2);
        *(v7 - 3) = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = v7 - 3;
        *(v7 - 2) = v11 + 16;
        *((_QWORD *)v11 + 2) = v7 - 4;
      }
      sub_BD6B50((unsigned __int8 *)v7, v32);
    }
    v21 = *(const char **)(a2 + 48);
    v22 = (const char **)(v7 + 6);
    v32[0] = v21;
    if ( v21 )
    {
      sub_B96E90((__int64)v32, (__int64)v21, 1);
      if ( v22 == v32 )
      {
        if ( v32[0] )
          sub_B91220((__int64)v32, (__int64)v32[0]);
        return (__int64)v7;
      }
      v27 = v7[6];
      if ( !v27 )
      {
LABEL_40:
        v28 = (unsigned __int8 *)v32[0];
        v7[6] = v32[0];
        if ( v28 )
          sub_B976B0((__int64)v32, v28, (__int64)(v7 + 6));
        return (__int64)v7;
      }
    }
    else
    {
      if ( v22 == v32 )
        return (__int64)v7;
      v27 = v7[6];
      if ( !v27 )
        return (__int64)v7;
    }
    sub_B91220((__int64)(v7 + 6), v27);
    goto LABEL_40;
  }
  return sub_2A89570(*a1, *((unsigned int *)a1 + 3), v4, a3, v5);
}
