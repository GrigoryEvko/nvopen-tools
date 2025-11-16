// Function: sub_1805E10
// Address: 0x1805e10
//
unsigned __int64 __fastcall sub_1805E10(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // r9
  __int64 v6; // rax
  unsigned __int8 *v7; // rsi
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 ***v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // r8
  __int64 *v13; // r11
  __int64 ****v14; // rbx
  __int64 v15; // rsi
  __int64 v16; // rax
  unsigned __int64 result; // rax
  __int64 v18; // rax
  _QWORD *v19; // r9
  _QWORD *v20; // r8
  __int64 *v21; // r11
  unsigned __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rdx
  unsigned __int8 *v26; // rsi
  _QWORD *v27; // [rsp+8h] [rbp-108h]
  _QWORD *v28; // [rsp+10h] [rbp-100h]
  __int64 *v29; // [rsp+10h] [rbp-100h]
  _QWORD *v30; // [rsp+10h] [rbp-100h]
  _QWORD *v31; // [rsp+10h] [rbp-100h]
  __int64 *v32; // [rsp+18h] [rbp-F8h]
  _QWORD *v33; // [rsp+18h] [rbp-F8h]
  _QWORD *v34; // [rsp+18h] [rbp-F8h]
  __int64 *v35; // [rsp+18h] [rbp-F8h]
  __int64 *v36; // [rsp+18h] [rbp-F8h]
  _QWORD *v37; // [rsp+20h] [rbp-F0h]
  _QWORD *v38; // [rsp+20h] [rbp-F0h]
  unsigned __int64 *v39; // [rsp+20h] [rbp-F0h]
  __int64 *v40; // [rsp+20h] [rbp-F0h]
  _QWORD *v41; // [rsp+20h] [rbp-F0h]
  _QWORD *v42; // [rsp+20h] [rbp-F0h]
  __int64 v43; // [rsp+38h] [rbp-D8h] BYREF
  _QWORD v44[2]; // [rsp+40h] [rbp-D0h] BYREF
  char v45; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v46; // [rsp+60h] [rbp-B0h]
  _QWORD v47[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v48; // [rsp+80h] [rbp-90h]
  unsigned __int8 *v49; // [rsp+90h] [rbp-80h] BYREF
  __int64 v50; // [rsp+98h] [rbp-78h]
  unsigned __int64 *v51; // [rsp+A0h] [rbp-70h]
  __int64 v52; // [rsp+A8h] [rbp-68h]
  __int64 v53; // [rsp+B0h] [rbp-60h]
  int v54; // [rsp+B8h] [rbp-58h]
  __int64 v55; // [rsp+C0h] [rbp-50h]
  __int64 v56; // [rsp+C8h] [rbp-48h]

  v3 = sub_16498A0(a2);
  v4 = *(_QWORD *)(a2 + 48);
  v49 = 0;
  v5 = v47;
  v52 = v3;
  v6 = *(_QWORD *)(a2 + 40);
  v53 = 0;
  v50 = v6;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v51 = (unsigned __int64 *)(a2 + 24);
  v47[0] = v4;
  if ( v4 )
  {
    sub_1623A60((__int64)v47, v4, 2);
    v5 = v47;
    if ( v49 )
    {
      sub_161E7C0((__int64)&v49, (__int64)v49);
      v7 = (unsigned __int8 *)v47[0];
      v5 = v47;
    }
    else
    {
      v7 = (unsigned __int8 *)v47[0];
    }
    v49 = v7;
    if ( v7 )
    {
      sub_1623210((__int64)v47, v7, (__int64)&v49);
      v5 = v47;
    }
  }
  v8 = a1[36];
  if ( *(_BYTE *)(a2 + 16) == 75 )
    v8 = a1[35];
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v9 = *(_QWORD *)(a2 - 8);
  else
    v9 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v10 = *(__int64 ****)v9;
  v11 = *(_QWORD *)(v9 + 24);
  v44[0] = v10;
  v12 = v44;
  v13 = (__int64 *)&v45;
  v14 = (__int64 ****)v44;
  v44[1] = v11;
  if ( *((_BYTE *)*v10 + 8) == 15 )
    goto LABEL_13;
  while ( v13 != (__int64 *)++v14 )
  {
    while ( 1 )
    {
      v10 = *v14;
      if ( *((_BYTE *)**v14 + 8) != 15 )
        break;
LABEL_13:
      v15 = a1[29];
      v46 = 257;
      if ( (__int64 **)v15 != *v10 )
      {
        if ( *((_BYTE *)v10 + 16) > 0x10u )
        {
          v29 = v13;
          v48 = 257;
          v33 = v12;
          v38 = v5;
          v18 = sub_15FDFF0((__int64)v10, v15, (__int64)v5, 0);
          v19 = v38;
          v20 = v33;
          v21 = v29;
          v10 = (__int64 ***)v18;
          if ( v50 )
          {
            v27 = v38;
            v39 = v51;
            sub_157E9D0(v50 + 40, v18);
            v19 = v27;
            v21 = v29;
            v22 = *v39;
            v23 = (unsigned __int64)v10[3] & 7;
            v10[4] = (__int64 **)v39;
            v20 = v33;
            v22 &= 0xFFFFFFFFFFFFFFF8LL;
            v10[3] = (__int64 **)(v22 | v23);
            *(_QWORD *)(v22 + 8) = v10 + 3;
            *v39 = *v39 & 7 | (unsigned __int64)(v10 + 3);
          }
          v30 = v19;
          v34 = v20;
          v40 = v21;
          sub_164B780((__int64)v10, v21);
          v13 = v40;
          v12 = v34;
          v5 = v30;
          if ( v49 )
          {
            v35 = v40;
            v41 = v12;
            v43 = (__int64)v49;
            sub_1623A60((__int64)&v43, (__int64)v49, 2);
            v24 = (__int64)v10[6];
            v12 = v41;
            v25 = (__int64)(v10 + 6);
            v13 = v35;
            v5 = v30;
            if ( v24 )
            {
              sub_161E7C0((__int64)(v10 + 6), v24);
              v5 = v30;
              v13 = v35;
              v12 = v41;
              v25 = (__int64)(v10 + 6);
            }
            v26 = (unsigned __int8 *)v43;
            v10[6] = (__int64 **)v43;
            if ( v26 )
            {
              v31 = v5;
              v36 = v13;
              v42 = v12;
              sub_1623210((__int64)&v43, v26, v25);
              v5 = v31;
              v13 = v36;
              v12 = v42;
            }
          }
        }
        else
        {
          v28 = v5;
          v32 = v13;
          v37 = v12;
          v16 = sub_15A4A70(v10, v15);
          v12 = v37;
          v13 = v32;
          v5 = v28;
          v10 = (__int64 ***)v16;
        }
      }
      *v14++ = v10;
      if ( v13 == (__int64 *)v14 )
        goto LABEL_17;
    }
  }
LABEL_17:
  v48 = 257;
  result = sub_1285290((__int64 *)&v49, *(_QWORD *)(v8 + 24), v8, (int)v12, 2, (__int64)v5, 0);
  if ( v49 )
    return sub_161E7C0((__int64)&v49, (__int64)v49);
  return result;
}
