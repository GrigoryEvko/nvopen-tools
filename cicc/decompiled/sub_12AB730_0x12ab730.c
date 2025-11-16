// Function: sub_12AB730
// Address: 0x12ab730
//
__int64 __fastcall sub_12AB730(__int64 a1, _QWORD *a2, __int64 a3, unsigned int a4)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  char *v10; // r12
  __int64 v11; // rsi
  char *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r15
  char *v15; // rax
  _QWORD *v16; // rdi
  char *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  char *v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 *v36; // [rsp+0h] [rbp-E0h]
  __int64 v37; // [rsp+0h] [rbp-E0h]
  __int64 v38; // [rsp+8h] [rbp-D8h]
  __int64 v39; // [rsp+8h] [rbp-D8h]
  __int64 v40; // [rsp+8h] [rbp-D8h]
  __int64 v41; // [rsp+8h] [rbp-D8h]
  __int64 v42; // [rsp+8h] [rbp-D8h]
  __int64 v44; // [rsp+18h] [rbp-C8h]
  __int64 *v45; // [rsp+20h] [rbp-C0h]
  unsigned __int64 *v46; // [rsp+38h] [rbp-A8h]
  __int64 v47; // [rsp+40h] [rbp-A0h]
  __int64 v48; // [rsp+48h] [rbp-98h]
  _QWORD v49[2]; // [rsp+50h] [rbp-90h] BYREF
  _BYTE v50[16]; // [rsp+60h] [rbp-80h] BYREF
  __int16 v51; // [rsp+70h] [rbp-70h]
  _QWORD v52[2]; // [rsp+80h] [rbp-60h] BYREF
  __int64 v53; // [rsp+90h] [rbp-50h]
  __int64 v54; // [rsp+98h] [rbp-48h]
  char *v55; // [rsp+A0h] [rbp-40h]

  v7 = sub_1643330(a2[5]);
  v48 = sub_1646BA0(v7, 3);
  v8 = sub_1643330(a2[5]);
  v47 = sub_1646BA0(v8, 1);
  v9 = sub_1643350(a2[5]);
  v51 = 257;
  v44 = sub_15A0680(v9, 0, 0);
  v45 = a2 + 6;
  v10 = sub_128F980((__int64)a2, a3);
  if ( v48 != *(_QWORD *)v10 )
  {
    if ( (unsigned __int8)v10[16] > 0x10u )
    {
      LOWORD(v53) = 257;
      v21 = sub_15FDFF0(v10, v48, v52, 0);
      v22 = a2[7];
      v10 = (char *)v21;
      if ( v22 )
      {
        v46 = (unsigned __int64 *)a2[8];
        sub_157E9D0(v22 + 40, v21);
        v23 = *v46;
        v24 = *((_QWORD *)v10 + 3) & 7LL;
        *((_QWORD *)v10 + 4) = v46;
        v23 &= 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)v10 + 3) = v23 | v24;
        *(_QWORD *)(v23 + 8) = v10 + 24;
        *v46 = *v46 & 7 | (unsigned __int64)(v10 + 24);
      }
      sub_164B780(v10, v50);
      v25 = a2[6];
      if ( v25 )
      {
        v49[0] = a2[6];
        sub_1623A60(v49, v25, 2);
        v26 = v10 + 48;
        if ( *((_QWORD *)v10 + 6) )
        {
          sub_161E7C0(v10 + 48);
          v26 = v10 + 48;
        }
        v27 = v49[0];
        *((_QWORD *)v10 + 6) = v49[0];
        if ( v27 )
          sub_1623210(v49, v27, v26);
      }
    }
    else
    {
      v10 = (char *)sub_15A4A70(v10, v48);
    }
  }
  v11 = *(_QWORD *)(a3 + 16);
  v51 = 257;
  v12 = sub_128F980((__int64)a2, v11);
  v13 = (__int64)v12;
  if ( v47 != *(_QWORD *)v12 )
  {
    if ( (unsigned __int8)v12[16] > 0x10u )
    {
      LOWORD(v53) = 257;
      v28 = sub_15FDFF0(v12, v47, v52, 0);
      v29 = a2[7];
      v30 = v28;
      if ( v29 )
      {
        v39 = v28;
        v36 = (__int64 *)a2[8];
        sub_157E9D0(v29 + 40, v28);
        v30 = v39;
        v31 = *(_QWORD *)(v39 + 24);
        v32 = *v36;
        *(_QWORD *)(v39 + 32) = v36;
        v32 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v39 + 24) = v32 | v31 & 7;
        *(_QWORD *)(v32 + 8) = v39 + 24;
        *v36 = *v36 & 7 | (v39 + 24);
      }
      v40 = v30;
      sub_164B780(v30, v50);
      v33 = a2[6];
      v13 = v40;
      if ( v33 )
      {
        v49[0] = a2[6];
        sub_1623A60(v49, v33, 2);
        v13 = v40;
        v34 = v40 + 48;
        if ( *(_QWORD *)(v40 + 48) )
        {
          v37 = v40;
          v41 = v40 + 48;
          sub_161E7C0(v41);
          v13 = v37;
          v34 = v41;
        }
        v35 = v49[0];
        *(_QWORD *)(v13 + 48) = v49[0];
        if ( v35 )
        {
          v42 = v13;
          sub_1623210(v49, v35, v34);
          v13 = v42;
        }
      }
    }
    else
    {
      v13 = sub_15A4A70(v12, v47);
    }
  }
  v38 = v13;
  v14 = sub_15A0680(v9, a4, 0);
  v15 = sub_128F980((__int64)a2, *(_QWORD *)(*(_QWORD *)(a3 + 16) + 16LL));
  v16 = (_QWORD *)a2[4];
  v17 = v15;
  v49[0] = v48;
  v49[1] = v47;
  v18 = sub_126A190(v16, 4163, (__int64)v49, 2u);
  v52[1] = v10;
  v54 = v14;
  v53 = v38;
  v52[0] = v44;
  v55 = v17;
  v51 = 257;
  v19 = sub_1285290(v45, *(_QWORD *)(v18 + 24), v18, (int)v52, 5, (__int64)v50, 0);
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v19;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
