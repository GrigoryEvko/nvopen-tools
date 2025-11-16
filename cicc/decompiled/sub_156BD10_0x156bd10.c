// Function: sub_156BD10
// Address: 0x156bd10
//
__int64 __fastcall sub_156BD10(__int64 *a1, __int64 a2, unsigned int a3)
{
  int v5; // eax
  __int64 v6; // rax
  _BYTE *v7; // r13
  __int64 v8; // rax
  __int64 v9; // r14
  _QWORD *v10; // r15
  __int64 v11; // r15
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  int v16; // r13d
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rax
  int v23; // eax
  unsigned int v24; // r11d
  __int64 v25; // rdi
  unsigned __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rsi
  _QWORD *v29; // rdx
  __int64 v30; // rsi
  unsigned int v31; // [rsp+8h] [rbp-98h]
  unsigned __int64 *v32; // [rsp+8h] [rbp-98h]
  __int64 v33; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+28h] [rbp-78h] BYREF
  char v37[16]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v38; // [rsp+40h] [rbp-60h]
  _BYTE v39[16]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v40; // [rsp+60h] [rbp-40h]

  v5 = *(_DWORD *)(a2 + 20);
  v38 = 257;
  v6 = v5 & 0xFFFFFFF;
  v7 = *(_BYTE **)(a2 - 24 * v6);
  v8 = 3 * (1 - v6);
  v9 = *(_QWORD *)(a2 + 8 * v8);
  if ( v7[16] > 0x10u || *(_BYTE *)(v9 + 16) > 0x10u )
  {
    v40 = 257;
    v20 = sub_1648A60(56, 2);
    v10 = (_QWORD *)v20;
    if ( v20 )
    {
      v33 = v20;
      v21 = *(_QWORD **)v7;
      if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 )
      {
        v31 = a3;
        v35 = v21[4];
        v22 = sub_1643320(*v21);
        v23 = sub_16463B0(v22, v35);
        v24 = v31;
      }
      else
      {
        v23 = sub_1643320(*v21);
        v24 = a3;
      }
      sub_15FEC10((_DWORD)v10, v23, 51, v24, (_DWORD)v7, v9, (__int64)v39, 0);
    }
    else
    {
      v33 = 0;
    }
    v25 = a1[1];
    if ( v25 )
    {
      v32 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v25 + 40, v10);
      v26 = *v32;
      v27 = v10[3] & 7LL;
      v10[4] = v32;
      v26 &= 0xFFFFFFFFFFFFFFF8LL;
      v10[3] = v26 | v27;
      *(_QWORD *)(v26 + 8) = v10 + 3;
      *v32 = *v32 & 7 | (unsigned __int64)(v10 + 3);
    }
    sub_164B780(v33, v37);
    v28 = *a1;
    if ( *a1 )
    {
      v36 = *a1;
      sub_1623A60(&v36, v28, 2);
      v29 = v10 + 6;
      if ( v10[6] )
      {
        sub_161E7C0(v10 + 6);
        v29 = v10 + 6;
      }
      v30 = v36;
      v10[6] = v36;
      if ( v30 )
        sub_1623210(&v36, v30, v29);
    }
  }
  else
  {
    v10 = (_QWORD *)sub_15A37B0(a3, v7, *(_QWORD *)(a2 + 8 * v8), 0);
  }
  v40 = 257;
  v11 = sub_156B790(a1, (__int64)v10, (__int64)v7, v9, (__int64)v39, 0);
  v12 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( *(char *)(a2 + 23) < 0 )
  {
    v13 = sub_1648A40(a2);
    v15 = v13 + v14;
    if ( *(char *)(a2 + 23) >= 0 )
    {
      if ( !(unsigned int)(v15 >> 4) )
        goto LABEL_10;
    }
    else
    {
      if ( !(unsigned int)((v15 - sub_1648A40(a2)) >> 4) )
        goto LABEL_10;
      if ( *(char *)(a2 + 23) < 0 )
      {
        v16 = *(_DWORD *)(sub_1648A40(a2) + 8);
        if ( *(char *)(a2 + 23) >= 0 )
          BUG();
        v17 = sub_1648A40(a2);
        v12 += v16 - *(_DWORD *)(v17 + v18 - 4);
        goto LABEL_10;
      }
    }
    BUG();
  }
LABEL_10:
  if ( v12 == 5 )
    return sub_156BB10(
             a1,
             *(_BYTE **)(a2 + 24 * (3LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
             v11,
             *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
  else
    return v11;
}
