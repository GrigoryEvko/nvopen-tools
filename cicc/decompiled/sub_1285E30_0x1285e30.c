// Function: sub_1285E30
// Address: 0x1285e30
//
__int64 __fastcall sub_1285E30(_QWORD *a1, __int64 a2, char a3)
{
  __int64 *v5; // r13
  _QWORD *v6; // r12
  _QWORD *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rsi
  _QWORD *v17; // rdx
  __int64 v18; // rsi
  unsigned __int64 *v19; // [rsp+10h] [rbp-90h]
  _QWORD *v20; // [rsp+18h] [rbp-88h] BYREF
  __int64 v21; // [rsp+28h] [rbp-78h] BYREF
  char v22[16]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v23; // [rsp+40h] [rbp-60h]
  _BYTE v24[16]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v25; // [rsp+60h] [rbp-40h]

  v5 = a1 + 6;
  v6 = (_QWORD *)a2;
  v7 = (_QWORD *)a1[4];
  v20 = (_QWORD *)a2;
  v23 = 257;
  v8 = v7[91];
  if ( v8 != *(_QWORD *)a2 )
  {
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      v25 = 257;
      v12 = sub_15FDBD0(47, a2, v8, v24, 0);
      v13 = a1[7];
      v6 = (_QWORD *)v12;
      if ( v13 )
      {
        v19 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v13 + 40, v12);
        v14 = *v19;
        v15 = v6[3] & 7LL;
        v6[4] = v19;
        v14 &= 0xFFFFFFFFFFFFFFF8LL;
        v6[3] = v14 | v15;
        *(_QWORD *)(v14 + 8) = v6 + 3;
        *v19 = *v19 & 7 | (unsigned __int64)(v6 + 3);
      }
      sub_164B780(v6, v22);
      v16 = a1[6];
      if ( v16 )
      {
        v21 = a1[6];
        sub_1623A60(&v21, v16, 2);
        v17 = v6 + 6;
        if ( v6[6] )
        {
          sub_161E7C0(v6 + 6);
          v17 = v6 + 6;
        }
        v18 = v21;
        v6[6] = v21;
        if ( v18 )
          sub_1623210(&v21, v18, v17);
      }
      v7 = (_QWORD *)a1[4];
    }
    else
    {
      v9 = sub_15A46C0(47, a2, v8, 0);
      v7 = (_QWORD *)a1[4];
      v6 = (_QWORD *)v9;
    }
  }
  v20 = v6;
  v25 = 257;
  v10 = sub_126A190(v7, 213 - ((unsigned int)(a3 == 0) - 1), 0, 0);
  return sub_1285290(v5, *(_QWORD *)(v10 + 24), v10, (int)&v20, 1, (__int64)v24, 0);
}
