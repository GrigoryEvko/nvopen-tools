// Function: sub_12AB0E0
// Address: 0x12ab0e0
//
__int64 __fastcall sub_12AB0E0(_QWORD *a1, int a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // [rsp+0h] [rbp-80h] BYREF
  __int64 v22; // [rsp+8h] [rbp-78h] BYREF
  _QWORD v23[2]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v24; // [rsp+20h] [rbp-60h]
  _BYTE v25[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v26; // [rsp+40h] [rbp-40h]

  v3 = a1[9];
  if ( a2 )
    v4 = sub_1643350(v3);
  else
    v4 = sub_1643320(v3);
  v5 = a1[9];
  v21 = v4;
  v26 = 257;
  v6 = sub_1643350(v5);
  v7 = sub_159C470(v6, a2, 0);
  v8 = (_QWORD *)a1[4];
  v23[0] = v7;
  v9 = sub_126A190(v8, 4275, (__int64)&v21, 1u);
  v10 = sub_1285290(a1 + 6, *(_QWORD *)(v9 + 24), v9, (int)v23, 1, (__int64)v25, 0);
  if ( !a2 )
  {
    v12 = a1[9];
    v24 = 257;
    v13 = sub_1643350(v12);
    if ( v13 != *(_QWORD *)v10 )
    {
      if ( *(_BYTE *)(v10 + 16) > 0x10u )
      {
        v26 = 257;
        v14 = sub_15FDBD0(37, v10, v13, v25, 0);
        v15 = a1[7];
        v10 = v14;
        if ( v15 )
        {
          v16 = (__int64 *)a1[8];
          sub_157E9D0(v15 + 40, v14);
          v17 = *(_QWORD *)(v10 + 24);
          v18 = *v16;
          *(_QWORD *)(v10 + 32) = v16;
          v18 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v10 + 24) = v18 | v17 & 7;
          *(_QWORD *)(v18 + 8) = v10 + 24;
          *v16 = *v16 & 7 | (v10 + 24);
        }
        sub_164B780(v10, v23);
        v19 = a1[6];
        if ( v19 )
        {
          v22 = a1[6];
          sub_1623A60(&v22, v19, 2);
          if ( *(_QWORD *)(v10 + 48) )
            sub_161E7C0(v10 + 48);
          v20 = v22;
          *(_QWORD *)(v10 + 48) = v22;
          if ( v20 )
            sub_1623210(&v22, v20, v10 + 48);
        }
      }
      else
      {
        return sub_15A46C0(37, v10, v13, 0);
      }
    }
  }
  return v10;
}
