// Function: sub_1A4F120
// Address: 0x1a4f120
//
void __fastcall sub_1A4F120(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r15
  __int64 *v14; // r12
  __int64 *v15; // rbx
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 v18; // rcx
  _QWORD *v19; // rax
  _QWORD *v20; // rbx
  unsigned __int64 *v21; // r12
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 *v26; // rsi
  int v27; // edi
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdx
  int v33; // eax
  bool v34; // al
  unsigned int v35; // [rsp+0h] [rbp-100h]
  __int64 v39; // [rsp+28h] [rbp-D8h]
  __int64 *v40; // [rsp+28h] [rbp-D8h]
  __int64 v41; // [rsp+28h] [rbp-D8h]
  __int64 v42; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v43[2]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v44; // [rsp+50h] [rbp-B0h]
  __int64 v45[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v46; // [rsp+70h] [rbp-90h]
  __int64 v47; // [rsp+80h] [rbp-80h] BYREF
  __int64 v48; // [rsp+88h] [rbp-78h]
  __int64 *v49; // [rsp+90h] [rbp-70h]
  __int64 v50; // [rsp+98h] [rbp-68h]
  __int64 v51; // [rsp+A0h] [rbp-60h]
  int v52; // [rsp+A8h] [rbp-58h]
  __int64 v53; // [rsp+B0h] [rbp-50h]
  __int64 v54; // [rsp+B8h] [rbp-48h]

  v11 = sub_157E9C0(a1);
  v48 = a1;
  v13 = *(_QWORD *)a2;
  v49 = (__int64 *)(a1 + 40);
  v14 = (__int64 *)(a2 + 8 * a3);
  v15 = (__int64 *)(a2 + 8);
  v47 = 0;
  v50 = v11;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  if ( v14 != (__int64 *)(a2 + 8) )
  {
    while ( 1 )
    {
      v16 = *v15;
      v44 = 257;
      if ( a4 )
      {
        if ( *(_BYTE *)(v16 + 16) > 0x10u )
          goto LABEL_35;
        v39 = v16;
        if ( !sub_1593BB0(v16, a2, v16, v12) )
        {
          v16 = v39;
          if ( *(_BYTE *)(v13 + 16) > 0x10u )
          {
LABEL_35:
            v27 = 27;
            v46 = 257;
            v26 = (__int64 *)v13;
            goto LABEL_28;
          }
          a2 = v39;
          v13 = sub_15A2D10((__int64 *)v13, v39, a7, a8, a9);
        }
LABEL_7:
        if ( v14 == ++v15 )
          break;
      }
      else
      {
        v17 = *(_BYTE *)(v16 + 16);
        if ( v17 > 0x10u )
          goto LABEL_27;
        if ( v17 == 13 )
        {
          a2 = *(unsigned int *)(v16 + 32);
          if ( (unsigned int)a2 <= 0x40 )
          {
            v12 = (unsigned int)(64 - a2);
            v34 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a2) == *(_QWORD *)(v16 + 24);
          }
          else
          {
            v35 = *(_DWORD *)(v16 + 32);
            v41 = v16;
            v33 = sub_16A58F0(v16 + 24);
            a2 = v35;
            v16 = v41;
            v34 = v35 == v33;
          }
          if ( v34 )
            goto LABEL_7;
        }
        if ( *(_BYTE *)(v13 + 16) > 0x10u )
        {
LABEL_27:
          v26 = (__int64 *)v13;
          v27 = 26;
          v46 = 257;
LABEL_28:
          v28 = sub_15FB440(v27, v26, v16, (__int64)v45, 0);
          v13 = v28;
          if ( v48 )
          {
            v40 = v49;
            sub_157E9D0(v48 + 40, v28);
            v29 = *v40;
            v30 = *(_QWORD *)(v13 + 24) & 7LL;
            *(_QWORD *)(v13 + 32) = v40;
            v29 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v13 + 24) = v29 | v30;
            *(_QWORD *)(v29 + 8) = v13 + 24;
            *v40 = *v40 & 7 | (v13 + 24);
          }
          sub_164B780(v13, v43);
          a2 = v47;
          if ( v47 )
          {
            v42 = v47;
            sub_1623A60((__int64)&v42, v47, 2);
            v31 = *(_QWORD *)(v13 + 48);
            v32 = v13 + 48;
            if ( v31 )
            {
              sub_161E7C0(v13 + 48, v31);
              v32 = v13 + 48;
            }
            a2 = v42;
            *(_QWORD *)(v13 + 48) = v42;
            if ( a2 )
              sub_1623210((__int64)&v42, (unsigned __int8 *)a2, v32);
          }
          goto LABEL_7;
        }
        a2 = v16;
        ++v15;
        v13 = sub_15A2CF0((__int64 *)v13, v16, a7, a8, a9);
        if ( v14 == v15 )
          break;
      }
    }
  }
  if ( a4 )
  {
    v18 = a5;
    a5 = a6;
    a6 = v18;
  }
  v46 = 257;
  v19 = sub_1648A60(56, 3u);
  v20 = v19;
  if ( v19 )
    sub_15F83E0((__int64)v19, a6, a5, v13, 0);
  if ( v48 )
  {
    v21 = (unsigned __int64 *)v49;
    sub_157E9D0(v48 + 40, (__int64)v20);
    v22 = v20[3];
    v23 = *v21;
    v20[4] = v21;
    v23 &= 0xFFFFFFFFFFFFFFF8LL;
    v20[3] = v23 | v22 & 7;
    *(_QWORD *)(v23 + 8) = v20 + 3;
    *v21 = *v21 & 7 | (unsigned __int64)(v20 + 3);
  }
  sub_164B780((__int64)v20, v45);
  if ( v47 )
  {
    v43[0] = v47;
    sub_1623A60((__int64)v43, v47, 2);
    v24 = v20[6];
    if ( v24 )
      sub_161E7C0((__int64)(v20 + 6), v24);
    v25 = (unsigned __int8 *)v43[0];
    v20[6] = v43[0];
    if ( v25 )
      sub_1623210((__int64)v43, v25, (__int64)(v20 + 6));
    if ( v47 )
      sub_161E7C0((__int64)&v47, v47);
  }
}
