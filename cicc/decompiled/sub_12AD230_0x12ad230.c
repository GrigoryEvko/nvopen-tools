// Function: sub_12AD230
// Address: 0x12ad230
//
__int64 __fastcall sub_12AD230(__int64 a1, _QWORD *a2, unsigned int a3, unsigned __int64 *a4, char a5)
{
  __int64 *v8; // r14
  __int64 v9; // r11
  unsigned __int64 *v10; // r10
  __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdi
  __int64 v21; // r15
  char *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  _QWORD *v29; // r15
  char *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdi
  unsigned __int64 *v33; // rbx
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rsi
  _QWORD *v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // [rsp+8h] [rbp-D8h]
  __int64 v40; // [rsp+10h] [rbp-D0h]
  __int64 v41; // [rsp+10h] [rbp-D0h]
  __int64 v42; // [rsp+10h] [rbp-D0h]
  __int64 v43; // [rsp+18h] [rbp-C8h]
  __int64 v44; // [rsp+18h] [rbp-C8h]
  int v45; // [rsp+2Ch] [rbp-B4h] BYREF
  __int64 v46; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+38h] [rbp-A8h] BYREF
  _QWORD v48[2]; // [rsp+40h] [rbp-A0h] BYREF
  char v49[16]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v50; // [rsp+60h] [rbp-80h]
  _QWORD v51[2]; // [rsp+70h] [rbp-70h] BYREF
  __int16 v52; // [rsp+80h] [rbp-60h]
  _BYTE v53[16]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v54; // [rsp+A0h] [rbp-40h]

  v8 = a2 + 6;
  v9 = *(_QWORD *)(a4[9] + 16);
  v10 = *(unsigned __int64 **)(v9 + 16);
  v11 = a2[4] + 8LL;
  v12 = *v10;
  if ( a5 )
  {
    v41 = *(_QWORD *)(v9 + 16);
    v39 = *(_QWORD *)(a4[9] + 16);
    v44 = v10[2];
    v19 = sub_127A030(v11, v12, 0);
    v20 = (_QWORD *)a2[4];
    v46 = v19;
    v21 = sub_126A190(v20, a3, (__int64)&v46, 1u);
    v48[0] = sub_128F980((__int64)a2, v39);
    v22 = sub_128F980((__int64)a2, v41);
    v54 = 257;
    v23 = *(_QWORD *)(v21 + 24);
    v48[1] = v22;
    v24 = sub_1285290(v8, v23, v21, (int)v48, 2, (__int64)v53, 0);
    v54 = 257;
    LODWORD(v51[0]) = 0;
    v42 = sub_12A9E60(v8, v24, (__int64)v51, 1, (__int64)v53);
    v25 = a2[4];
    v52 = 257;
    v26 = sub_127A030(v25 + 8, *a4, 0);
    v50 = 257;
    v27 = v26;
    v45 = 1;
    v28 = sub_12A9E60(v8, v24, (__int64)&v45, 1, (__int64)v49);
    v29 = (_QWORD *)v28;
    if ( v27 != *(_QWORD *)v28 )
    {
      if ( *(_BYTE *)(v28 + 16) > 0x10u )
      {
        v54 = 257;
        v31 = sub_15FDBD0(37, v28, v27, v53, 0);
        v32 = a2[7];
        v29 = (_QWORD *)v31;
        if ( v32 )
        {
          v33 = (unsigned __int64 *)a2[8];
          sub_157E9D0(v32 + 40, v31);
          v34 = v29[3];
          v35 = *v33;
          v29[4] = v33;
          v35 &= 0xFFFFFFFFFFFFFFF8LL;
          v29[3] = v35 | v34 & 7;
          *(_QWORD *)(v35 + 8) = v29 + 3;
          *v33 = *v33 & 7 | (unsigned __int64)(v29 + 3);
        }
        sub_164B780(v29, v51);
        v36 = a2[6];
        if ( v36 )
        {
          v47 = a2[6];
          sub_1623A60(&v47, v36, 2);
          v37 = v29 + 6;
          if ( v29[6] )
          {
            sub_161E7C0(v29 + 6);
            v37 = v29 + 6;
          }
          v38 = v47;
          v29[6] = v47;
          if ( v38 )
            sub_1623210(&v47, v38, v37);
        }
      }
      else
      {
        v29 = (_QWORD *)sub_15A46C0(37, v28, v27, 0);
      }
    }
    v30 = sub_128F980((__int64)a2, v44);
    sub_12A8F50(v8, (__int64)v29, (__int64)v30, 0);
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_DWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v42;
    *(_DWORD *)(a1 + 16) = 0;
  }
  else
  {
    v40 = *(_QWORD *)(v9 + 16);
    v43 = *(_QWORD *)(a4[9] + 16);
    v13 = sub_127A030(v11, v12, 0);
    v14 = (_QWORD *)a2[4];
    v46 = v13;
    v15 = sub_126A190(v14, a3, (__int64)&v46, 1u);
    v51[0] = sub_128F980((__int64)a2, v43);
    v51[1] = sub_128F980((__int64)a2, v40);
    v54 = 257;
    v16 = sub_1285290(v8, *(_QWORD *)(v15 + 24), v15, (int)v51, 2, (__int64)v53, 0);
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_QWORD *)a1 = v16;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
  }
  return a1;
}
