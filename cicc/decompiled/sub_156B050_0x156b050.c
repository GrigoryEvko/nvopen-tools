// Function: sub_156B050
// Address: 0x156b050
//
__int64 __fastcall sub_156B050(__int64 *a1, _BYTE *a2, unsigned int a3)
{
  _QWORD *v3; // r12
  __int64 v4; // r14
  unsigned int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r15
  unsigned int v9; // esi
  unsigned int v10; // eax
  unsigned int v11; // edx
  __int64 v12; // rcx
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rsi
  _QWORD *v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int64 *v26; // [rsp+10h] [rbp-170h]
  __int64 v28; // [rsp+28h] [rbp-158h] BYREF
  _QWORD v29[2]; // [rsp+30h] [rbp-150h] BYREF
  __int16 v30; // [rsp+40h] [rbp-140h]
  _DWORD v31[4]; // [rsp+50h] [rbp-130h] BYREF
  __int16 v32; // [rsp+60h] [rbp-120h]

  v3 = a2;
  v4 = *(_QWORD *)a2;
  v5 = 8 * *(_DWORD *)(*(_QWORD *)a2 + 32LL);
  v6 = sub_1643330(a1[3]);
  v7 = sub_16463B0(v6, v5);
  v30 = 259;
  v29[0] = "cast";
  if ( v7 != *(_QWORD *)a2 )
  {
    if ( a2[16] > 0x10u )
    {
      v32 = 257;
      v3 = (_QWORD *)sub_15FDBD0(47, a2, v7, v31, 0);
      v14 = a1[1];
      if ( v14 )
      {
        v26 = (unsigned __int64 *)a1[2];
        sub_157E9D0(v14 + 40, v3);
        v15 = *v26;
        v16 = v3[3] & 7LL;
        v3[4] = v26;
        v15 &= 0xFFFFFFFFFFFFFFF8LL;
        v3[3] = v15 | v16;
        *(_QWORD *)(v15 + 8) = v3 + 3;
        *v26 = *v26 & 7 | (unsigned __int64)(v3 + 3);
      }
      sub_164B780(v3, v29);
      v17 = *a1;
      if ( *a1 )
      {
        v28 = *a1;
        sub_1623A60(&v28, v17, 2);
        v18 = v3 + 6;
        if ( v3[6] )
        {
          sub_161E7C0(v3 + 6);
          v18 = v3 + 6;
        }
        v19 = v28;
        v3[6] = v28;
        if ( v19 )
          sub_1623210(&v28, v19, v18);
      }
    }
    else
    {
      v3 = (_QWORD *)sub_15A46C0(47, a2, v7, 0);
    }
  }
  v8 = sub_15A06D0(v7);
  if ( a3 <= 0xF )
  {
    if ( v5 )
    {
      v9 = a3 - v5;
      do
      {
        v10 = v5 - a3;
        do
        {
          v11 = 16 - v5 + v10;
          v12 = v9 + v10;
          if ( v5 <= v10 )
            v11 = v10;
          ++v10;
          v31[v12] = v5 - a3 + v9 + v11;
        }
        while ( v10 != v5 - a3 + 16 );
        v9 += 16;
      }
      while ( a3 != v9 );
    }
    v30 = 257;
    v8 = sub_156A7D0(a1, v8, (__int64)v3, (__int64)v31, v5, (__int64)v29);
  }
  v29[0] = "cast";
  v30 = 259;
  if ( v4 != *(_QWORD *)v8 )
  {
    if ( *(_BYTE *)(v8 + 16) > 0x10u )
    {
      v32 = 257;
      v8 = sub_15FDBD0(47, v8, v4, v31, 0);
      v20 = a1[1];
      if ( v20 )
      {
        v21 = (__int64 *)a1[2];
        sub_157E9D0(v20 + 40, v8);
        v22 = *(_QWORD *)(v8 + 24);
        v23 = *v21;
        *(_QWORD *)(v8 + 32) = v21;
        v23 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v23 | v22 & 7;
        *(_QWORD *)(v23 + 8) = v8 + 24;
        *v21 = *v21 & 7 | (v8 + 24);
      }
      sub_164B780(v8, v29);
      v24 = *a1;
      if ( *a1 )
      {
        v28 = *a1;
        sub_1623A60(&v28, v24, 2);
        if ( *(_QWORD *)(v8 + 48) )
          sub_161E7C0(v8 + 48);
        v25 = v28;
        *(_QWORD *)(v8 + 48) = v28;
        if ( v25 )
          sub_1623210(&v28, v25, v8 + 48);
      }
    }
    else
    {
      return sub_15A46C0(47, v8, v4, 0);
    }
  }
  return v8;
}
