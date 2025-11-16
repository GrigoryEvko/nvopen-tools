// Function: sub_156B3E0
// Address: 0x156b3e0
//
__int64 __fastcall sub_156B3E0(__int64 *a1, _BYTE *a2, unsigned int a3)
{
  _QWORD *v3; // r15
  __int64 *v4; // r12
  __int64 v6; // r13
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r14
  int v11; // r11d
  unsigned int v12; // eax
  unsigned int v13; // edx
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  _QWORD *v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 *v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rsi
  unsigned __int64 *v27; // [rsp+10h] [rbp-170h]
  unsigned int v28; // [rsp+18h] [rbp-168h]
  __int64 v29; // [rsp+28h] [rbp-158h] BYREF
  _QWORD v30[2]; // [rsp+30h] [rbp-150h] BYREF
  __int16 v31; // [rsp+40h] [rbp-140h]
  _DWORD v32[4]; // [rsp+50h] [rbp-130h] BYREF
  __int16 v33; // [rsp+60h] [rbp-120h]

  v3 = a2;
  v4 = a1;
  v6 = *(_QWORD *)a2;
  v7 = 8 * *(_DWORD *)(*(_QWORD *)a2 + 32LL);
  v28 = v7;
  v8 = sub_1643330(a1[3]);
  v9 = sub_16463B0(v8, v7);
  v31 = 259;
  v30[0] = "cast";
  if ( v9 != *(_QWORD *)a2 )
  {
    if ( a2[16] > 0x10u )
    {
      v33 = 257;
      v3 = (_QWORD *)sub_15FDBD0(47, a2, v9, v32, 0);
      v15 = a1[1];
      if ( v15 )
      {
        v27 = (unsigned __int64 *)a1[2];
        sub_157E9D0(v15 + 40, v3);
        v16 = *v27;
        v17 = v3[3] & 7LL;
        v3[4] = v27;
        v16 &= 0xFFFFFFFFFFFFFFF8LL;
        v3[3] = v16 | v17;
        *(_QWORD *)(v16 + 8) = v3 + 3;
        *v27 = *v27 & 7 | (unsigned __int64)(v3 + 3);
      }
      sub_164B780(v3, v30);
      v18 = *a1;
      if ( *a1 )
      {
        v29 = *a1;
        sub_1623A60(&v29, v18, 2);
        v19 = v3 + 6;
        if ( v3[6] )
        {
          sub_161E7C0(v3 + 6);
          v19 = v3 + 6;
        }
        v20 = v29;
        v3[6] = v29;
        if ( v20 )
          sub_1623210(&v29, v20, v19);
      }
    }
    else
    {
      v3 = (_QWORD *)sub_15A46C0(47, a2, v9, 0);
    }
  }
  v10 = sub_15A06D0(v9);
  if ( a3 <= 0xF )
  {
    if ( v28 )
    {
      v11 = -a3;
      do
      {
        v12 = a3 + 1;
        v13 = a3;
        while ( 1 )
        {
          v32[v11 - 1 + v12] = a3 + v11 + v13;
          if ( a3 + 16 == v12 )
            break;
          v13 = v28 - 16 + v12;
          if ( v12 <= 0xF )
            v13 = v12;
          ++v12;
        }
        v11 += 16;
      }
      while ( v11 != v28 - a3 );
      v4 = a1;
    }
    v31 = 257;
    v10 = sub_156A7D0(v4, (__int64)v3, v10, (__int64)v32, v28, (__int64)v30);
  }
  v30[0] = "cast";
  v31 = 259;
  if ( v6 != *(_QWORD *)v10 )
  {
    if ( *(_BYTE *)(v10 + 16) > 0x10u )
    {
      v33 = 257;
      v10 = sub_15FDBD0(47, v10, v6, v32, 0);
      v21 = v4[1];
      if ( v21 )
      {
        v22 = (__int64 *)v4[2];
        sub_157E9D0(v21 + 40, v10);
        v23 = *(_QWORD *)(v10 + 24);
        v24 = *v22;
        *(_QWORD *)(v10 + 32) = v22;
        v24 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v10 + 24) = v24 | v23 & 7;
        *(_QWORD *)(v24 + 8) = v10 + 24;
        *v22 = *v22 & 7 | (v10 + 24);
      }
      sub_164B780(v10, v30);
      v25 = *v4;
      if ( *v4 )
      {
        v29 = *v4;
        sub_1623A60(&v29, v25, 2);
        if ( *(_QWORD *)(v10 + 48) )
          sub_161E7C0(v10 + 48);
        v26 = v29;
        *(_QWORD *)(v10 + 48) = v29;
        if ( v26 )
          sub_1623210(&v29, v26, v10 + 48);
      }
    }
    else
    {
      return sub_15A46C0(47, v10, v6, 0);
    }
  }
  return v10;
}
