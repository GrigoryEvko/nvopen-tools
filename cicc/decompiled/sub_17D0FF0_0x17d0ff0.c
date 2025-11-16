// Function: sub_17D0FF0
// Address: 0x17d0ff0
//
unsigned __int64 __fastcall sub_17D0FF0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  _QWORD *v3; // rax
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 **v13; // rax
  __int64 *v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // r14
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 **v21; // r14
  __int64 v22; // rax
  __int64 **v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rdx
  unsigned __int8 *v28; // rsi
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int8 *v33; // rsi
  __int64 v34; // [rsp+8h] [rbp-168h]
  __int64 *v35; // [rsp+10h] [rbp-160h]
  _QWORD *v36; // [rsp+20h] [rbp-150h]
  __int64 v37; // [rsp+28h] [rbp-148h]
  __int64 *v38; // [rsp+28h] [rbp-148h]
  __int64 *v39; // [rsp+28h] [rbp-148h]
  __int64 v40; // [rsp+38h] [rbp-138h] BYREF
  __int64 v41; // [rsp+40h] [rbp-130h] BYREF
  __int16 v42; // [rsp+50h] [rbp-120h]
  __int64 v43; // [rsp+60h] [rbp-110h] BYREF
  __int16 v44; // [rsp+70h] [rbp-100h]
  _BYTE v45[16]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v46; // [rsp+90h] [rbp-E0h]
  __int64 v47[10]; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v48; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v49; // [rsp+F8h] [rbp-78h]
  __int64 *v50; // [rsp+100h] [rbp-70h]
  _QWORD *v51; // [rsp+108h] [rbp-68h]

  v1 = sub_157ED20(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 480LL));
  sub_17CE510((__int64)v47, v1, 0, 0, 0);
  v2 = *(_QWORD *)(a1 + 16);
  LOWORD(v50) = 257;
  v3 = sub_156E5B0(v47, *(_QWORD *)(v2 + 232), (__int64)&v48);
  *(_QWORD *)(a1 + 40) = v3;
  v4 = (__int64)v3;
  v5 = *(_QWORD *)(a1 + 16);
  LOWORD(v50) = 257;
  v6 = sub_15A0680(*(_QWORD *)(v5 + 176), 0, 0);
  v35 = (__int64 *)sub_12899C0(v47, v6, v4, (__int64)&v48, 0, 0);
  if ( *(_DWORD *)(a1 + 56) )
  {
    v8 = *(_QWORD *)(a1 + 16);
    LOWORD(v50) = 257;
    v9 = (_QWORD *)sub_1643330(*(_QWORD **)(v8 + 168));
    v10 = sub_17CEAE0(v47, v9, (__int64)v35, &v48);
    *(_QWORD *)(a1 + 32) = v10;
    sub_15E7430(v47, v10, 8u, *(_QWORD **)(*(_QWORD *)(a1 + 16) + 224LL), 8u, v35, 0, 0, 0, 0, 0);
    v34 = *(unsigned int *)(a1 + 56);
    if ( *(_DWORD *)(a1 + 56) )
    {
      v11 = 0;
      do
      {
        v16 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v11);
        v17 = *(_QWORD *)(v16 + 32);
        if ( v17 == *(_QWORD *)(v16 + 40) + 40LL || !v17 )
          v18 = 0;
        else
          v18 = v17 - 24;
        sub_17CE510((__int64)&v48, v18, 0, 0, 0);
        v12 = *(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
        v19 = *(_QWORD *)(a1 + 16);
        v44 = 257;
        v20 = (__int64 *)sub_1647230(*(_QWORD **)(v19 + 168), 0);
        v21 = (__int64 **)sub_1646BA0(v20, 0);
        v22 = *(_QWORD *)(a1 + 16);
        v42 = 257;
        v23 = *(__int64 ***)(v22 + 176);
        v13 = *(__int64 ***)v12;
        if ( v23 != *(__int64 ***)v12 )
        {
          if ( *(_BYTE *)(v12 + 16) <= 0x10u )
          {
            v12 = sub_15A46C0(45, (__int64 ***)v12, v23, 0);
            v13 = *(__int64 ***)v12;
          }
          else
          {
            v46 = 257;
            v12 = sub_15FDBD0(45, v12, (__int64)v23, (__int64)v45, 0);
            if ( v49 )
            {
              v38 = v50;
              sub_157E9D0(v49 + 40, v12);
              v24 = *v38;
              v25 = *(_QWORD *)(v12 + 24) & 7LL;
              *(_QWORD *)(v12 + 32) = v38;
              v24 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v12 + 24) = v24 | v25;
              *(_QWORD *)(v24 + 8) = v12 + 24;
              *v38 = *v38 & 7 | (v12 + 24);
            }
            sub_164B780(v12, &v41);
            if ( v48 )
            {
              v40 = v48;
              sub_1623A60((__int64)&v40, v48, 2);
              v26 = *(_QWORD *)(v12 + 48);
              v27 = v12 + 48;
              if ( v26 )
              {
                sub_161E7C0(v12 + 48, v26);
                v27 = v12 + 48;
              }
              v28 = (unsigned __int8 *)v40;
              *(_QWORD *)(v12 + 48) = v40;
              if ( v28 )
                sub_1623210((__int64)&v40, v28, v27);
            }
            v13 = *(__int64 ***)v12;
          }
        }
        if ( v21 != v13 )
        {
          if ( *(_BYTE *)(v12 + 16) > 0x10u )
          {
            v46 = 257;
            v12 = sub_15FDBD0(46, v12, (__int64)v21, (__int64)v45, 0);
            if ( v49 )
            {
              v39 = v50;
              sub_157E9D0(v49 + 40, v12);
              v29 = *v39;
              v30 = *(_QWORD *)(v12 + 24) & 7LL;
              *(_QWORD *)(v12 + 32) = v39;
              v29 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v12 + 24) = v29 | v30;
              *(_QWORD *)(v29 + 8) = v12 + 24;
              *v39 = *v39 & 7 | (v12 + 24);
            }
            sub_164B780(v12, &v43);
            if ( v48 )
            {
              v40 = v48;
              sub_1623A60((__int64)&v40, v48, 2);
              v31 = *(_QWORD *)(v12 + 48);
              v32 = v12 + 48;
              if ( v31 )
              {
                sub_161E7C0(v12 + 48, v31);
                v32 = v12 + 48;
              }
              v33 = (unsigned __int8 *)v40;
              *(_QWORD *)(v12 + 48) = v40;
              if ( v33 )
                sub_1623210((__int64)&v40, v33, v32);
            }
          }
          else
          {
            v12 = sub_15A46C0(46, (__int64 ***)v12, v21, 0);
          }
        }
        v46 = 257;
        v36 = sub_156E5B0(&v48, v12, (__int64)v45);
        v37 = *(_QWORD *)(a1 + 24);
        v14 = (__int64 *)sub_1643330(v51);
        v15 = (_QWORD *)sub_17CFB40(v37, (__int64)v36, &v48, v14, 8u);
        sub_15E7430(&v48, v15, 8u, *(_QWORD **)(a1 + 32), 8u, v35, 0, 0, 0, 0, 0);
        if ( v48 )
          sub_161E7C0((__int64)&v48, v48);
        ++v11;
      }
      while ( v11 != v34 );
    }
  }
  return sub_17CD270(v47);
}
