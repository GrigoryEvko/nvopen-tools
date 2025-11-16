// Function: sub_17F67F0
// Address: 0x17f67f0
//
__int64 __fastcall sub_17F67F0(__int64 a1, __int64 *a2, const char *a3, __int64 a4, const char *a5)
{
  __int64 v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // r8
  _QWORD *v12; // r12
  _QWORD *v13; // rbx
  __int64 v14; // rax
  int v15; // eax
  int v16; // r9d
  __int64 v17; // r12
  __int64 v19; // rax
  unsigned __int64 *v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned __int8 *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r8
  unsigned __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  unsigned __int8 *v32; // rsi
  __int64 v33; // [rsp+10h] [rbp-130h]
  unsigned __int64 *v34; // [rsp+10h] [rbp-130h]
  __int64 v35; // [rsp+10h] [rbp-130h]
  __int64 v36; // [rsp+10h] [rbp-130h]
  __int64 v37; // [rsp+20h] [rbp-120h]
  __int64 v38; // [rsp+48h] [rbp-F8h] BYREF
  _QWORD v39[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v40; // [rsp+60h] [rbp-E0h] BYREF
  __int16 v41; // [rsp+70h] [rbp-D0h]
  __int64 v42[2]; // [rsp+80h] [rbp-C0h] BYREF
  __int16 v43; // [rsp+90h] [rbp-B0h]
  _QWORD v44[2]; // [rsp+A0h] [rbp-A0h] BYREF
  __int16 v45; // [rsp+B0h] [rbp-90h]
  __int64 v46; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v47; // [rsp+C8h] [rbp-78h]
  unsigned __int64 *v48; // [rsp+D0h] [rbp-70h]
  __int64 v49; // [rsp+D8h] [rbp-68h]
  __int64 v50; // [rsp+E0h] [rbp-60h]
  int v51; // [rsp+E8h] [rbp-58h]
  __int64 v52; // [rsp+F0h] [rbp-50h]
  __int64 v53; // [rsp+F8h] [rbp-48h]

  v8 = *a2;
  v46 = 0;
  v48 = 0;
  v49 = v8;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v47 = 0;
  v9 = sub_17F6490(a1, (__int64)a2, a5, a4);
  v41 = 257;
  v11 = (__int64)v10;
  v12 = v9;
  v13 = v10;
  if ( a4 != *v9 )
  {
    v33 = (__int64)v10;
    if ( *((_BYTE *)v9 + 16) > 0x10u )
    {
      v45 = 257;
      v26 = sub_15FDFF0((__int64)v9, a4, (__int64)v44, 0);
      v27 = v33;
      v12 = (_QWORD *)v26;
      if ( v47 )
      {
        v37 = v33;
        v34 = v48;
        sub_157E9D0(v47 + 40, v26);
        v27 = v37;
        v28 = *v34;
        v29 = v12[3] & 7LL;
        v12[4] = v34;
        v28 &= 0xFFFFFFFFFFFFFFF8LL;
        v12[3] = v28 | v29;
        *(_QWORD *)(v28 + 8) = v12 + 3;
        *v34 = *v34 & 7 | (unsigned __int64)(v12 + 3);
      }
      v35 = v27;
      sub_164B780((__int64)v12, &v40);
      v11 = v35;
      if ( v46 )
      {
        v42[0] = v46;
        sub_1623A60((__int64)v42, v46, 2);
        v30 = v12[6];
        v31 = (__int64)(v12 + 6);
        v11 = v35;
        if ( v30 )
        {
          sub_161E7C0((__int64)(v12 + 6), v30);
          v11 = v35;
          v31 = (__int64)(v12 + 6);
        }
        v32 = (unsigned __int8 *)v42[0];
        v12[6] = v42[0];
        if ( v32 )
        {
          v36 = v11;
          sub_1623210((__int64)v42, v32, v31);
          v11 = v36;
        }
      }
    }
    else
    {
      v14 = sub_15A4A70((__int64 ***)v9, a4);
      v11 = v33;
      v12 = (_QWORD *)v14;
    }
  }
  v39[0] = v12;
  v43 = 257;
  if ( a4 != *(_QWORD *)v11 )
  {
    if ( *(_BYTE *)(v11 + 16) > 0x10u )
    {
      v45 = 257;
      v19 = sub_15FDFF0(v11, a4, (__int64)v44, 0);
      v13 = (_QWORD *)v19;
      if ( v47 )
      {
        v20 = v48;
        sub_157E9D0(v47 + 40, v19);
        v21 = v13[3];
        v22 = *v20;
        v13[4] = v20;
        v22 &= 0xFFFFFFFFFFFFFFF8LL;
        v13[3] = v22 | v21 & 7;
        *(_QWORD *)(v22 + 8) = v13 + 3;
        *v20 = *v20 & 7 | (unsigned __int64)(v13 + 3);
      }
      sub_164B780((__int64)v13, v42);
      if ( v46 )
      {
        v38 = v46;
        sub_1623A60((__int64)&v38, v46, 2);
        v23 = v13[6];
        v24 = (__int64)(v13 + 6);
        if ( v23 )
        {
          sub_161E7C0((__int64)(v13 + 6), v23);
          v24 = (__int64)(v13 + 6);
        }
        v25 = (unsigned __int8 *)v38;
        v13[6] = v38;
        if ( v25 )
          sub_1623210((__int64)&v38, v25, v24);
      }
    }
    else
    {
      v13 = (_QWORD *)sub_15A4A70((__int64 ***)v11, a4);
    }
  }
  v39[1] = v13;
  v44[0] = a4;
  v44[1] = a4;
  v15 = strlen(a3);
  v17 = sub_1B281E0(
          (_DWORD)a2,
          (unsigned int)"sancov.module_ctor",
          18,
          (_DWORD)a3,
          v15,
          v16,
          (__int64)v44,
          2,
          (__int64)v39,
          2,
          0,
          0);
  if ( *(_DWORD *)(a1 + 428) == 3 )
  {
    sub_1B28000(a2, v17, 2, 0);
  }
  else
  {
    *(_QWORD *)(v17 + 48) = sub_1633B90((__int64)a2, "sancov.module_ctor", 0x12u);
    sub_1B28000(a2, v17, 2, v17);
  }
  if ( v46 )
    sub_161E7C0((__int64)&v46, v46);
  return v17;
}
