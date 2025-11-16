// Function: sub_1289D20
// Address: 0x1289d20
//
__int64 __fastcall sub_1289D20(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v6; // dl
  char v7; // al
  unsigned __int8 v8; // al
  __int64 *v9; // rbx
  __int64 v10; // r12
  int v11; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // r14d
  __int64 v16; // rdi
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rsi
  bool v22; // cc
  __int64 *v23; // rbx
  __int64 v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 *v29; // r13
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 *v36; // r12
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // [rsp+0h] [rbp-B0h]
  __int64 v45; // [rsp+8h] [rbp-A8h]
  __int64 v46; // [rsp+10h] [rbp-A0h]
  __int64 v47; // [rsp+18h] [rbp-98h]
  __int64 v48; // [rsp+20h] [rbp-90h]
  __int64 v49; // [rsp+20h] [rbp-90h]
  __int64 v50; // [rsp+20h] [rbp-90h]
  __int64 v51; // [rsp+28h] [rbp-88h]
  char v52; // [rsp+28h] [rbp-88h]
  __int64 v53; // [rsp+38h] [rbp-78h] BYREF
  _QWORD v54[2]; // [rsp+40h] [rbp-70h] BYREF
  char v55; // [rsp+50h] [rbp-60h]
  char v56; // [rsp+51h] [rbp-5Fh]
  __int64 v57; // [rsp+60h] [rbp-50h] BYREF
  __int64 v58; // [rsp+68h] [rbp-48h] BYREF
  __int64 v59; // [rsp+70h] [rbp-40h]

  v6 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  v7 = v6;
  if ( v6 == 16 )
    v7 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
  if ( (unsigned __int8)(v7 - 1) <= 5u )
  {
    v8 = *(_BYTE *)(a2 + 16);
    if ( v8 == 14 )
    {
      if ( v6 != 2 )
      {
        v56 = 1;
        v9 = (__int64 *)a1[1];
        v54[0] = "div";
        v55 = 3;
        goto LABEL_7;
      }
      v51 = sub_1698280(a4);
      sub_169D3F0(v54, 1.0);
      sub_169E320(&v58, v54, v51);
      sub_1698460(v54);
      sub_16A3360(&v57, *(_QWORD *)(a2 + 32), 0, v54);
      v52 = sub_1594120(a2, &v57);
      v24 = sub_16982C0();
      if ( v58 == v24 )
      {
        v45 = v59;
        if ( v59 )
        {
          v38 = v59 + 32LL * *(_QWORD *)(v59 - 8);
          if ( v59 != v38 )
          {
            do
            {
              v38 -= 32;
              if ( v24 == *(_QWORD *)(v38 + 8) )
              {
                v39 = *(_QWORD *)(v38 + 16);
                v44 = v39;
                if ( v39 )
                {
                  v40 = 32LL * *(_QWORD *)(v39 - 8) + v39;
                  if ( *(_QWORD *)(v38 + 16) != v40 )
                  {
                    do
                    {
                      v40 -= 32;
                      if ( v24 == *(_QWORD *)(v40 + 8) )
                      {
                        v41 = *(_QWORD *)(v40 + 16);
                        if ( v41 )
                        {
                          v42 = 32LL * *(_QWORD *)(v41 - 8);
                          v43 = v41 + v42;
                          if ( v41 != v41 + v42 )
                          {
                            do
                            {
                              v46 = v41;
                              v47 = v40;
                              v49 = v43 - 32;
                              sub_127D120((_QWORD *)(v43 - 24));
                              v43 = v49;
                              v41 = v46;
                              v40 = v47;
                            }
                            while ( v46 != v49 );
                          }
                          v50 = v40;
                          j_j_j___libc_free_0_0(v41 - 8);
                          v40 = v50;
                        }
                      }
                      else
                      {
                        v48 = v40;
                        sub_1698460(v40 + 8);
                        v40 = v48;
                      }
                    }
                    while ( v44 != v40 );
                  }
                  j_j_j___libc_free_0_0(v44 - 8);
                }
              }
              else
              {
                sub_1698460(v38 + 8);
              }
            }
            while ( v45 != v38 );
          }
          j_j_j___libc_free_0_0(v45 - 8);
        }
      }
      else
      {
        sub_1698460(&v58);
      }
      if ( v52 )
        goto LABEL_16;
    }
    else if ( v6 != 2 )
    {
      goto LABEL_17;
    }
    if ( unk_4D0451C )
    {
      v34 = unk_4D04518 == 0 ? 3847 : 3849;
    }
    else
    {
      if ( !unk_4D04518 )
      {
LABEL_16:
        v8 = *(_BYTE *)(a2 + 16);
LABEL_17:
        v56 = 1;
        v9 = (__int64 *)a1[1];
        v54[0] = "div";
        v55 = 3;
        if ( v8 > 0x10u )
        {
LABEL_18:
          LOWORD(v59) = 257;
          v13 = sub_15FB440(19, a2, a3, &v57, 0);
          v14 = v9[4];
          v15 = *((_DWORD *)v9 + 10);
          v10 = v13;
          if ( v14 )
            sub_1625C10(v13, 3, v14);
          sub_15F2440(v10, v15);
          v16 = v9[1];
          if ( v16 )
          {
            v17 = (__int64 *)v9[2];
            sub_157E9D0(v16 + 40, v10);
            v18 = *(_QWORD *)(v10 + 24);
            v19 = *v17;
            *(_QWORD *)(v10 + 32) = v17;
            v19 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v10 + 24) = v19 | v18 & 7;
            *(_QWORD *)(v19 + 8) = v10 + 24;
            *v17 = *v17 & 7 | (v10 + 24);
          }
          sub_164B780(v10, v54);
          v20 = *v9;
          if ( *v9 )
          {
            v57 = *v9;
            sub_1623A60(&v57, v20, 2);
            if ( *(_QWORD *)(v10 + 48) )
              sub_161E7C0(v10 + 48);
            v21 = v57;
            *(_QWORD *)(v10 + 48) = v57;
            if ( v21 )
              sub_1623210(&v57, v21, v10 + 48);
          }
LABEL_9:
          if ( !unk_4D04700 )
            return v10;
          goto LABEL_10;
        }
LABEL_7:
        if ( *(_BYTE *)(a3 + 16) <= 0x10u )
        {
          v10 = sub_15A2A30(19, a2, a3, 0, 0);
          if ( v10 )
            goto LABEL_9;
        }
        goto LABEL_18;
      }
      v34 = 3849;
    }
    v35 = *a1;
    v54[0] = a2;
    v54[1] = a3;
    v36 = (__int64 *)a1[1];
    LOWORD(v59) = 257;
    v37 = sub_15E26F0(**(_QWORD **)(v35 + 32), v34, 0, 0);
    v10 = sub_1285290(v36, *(_QWORD *)(v37 + 24), v37, (int)v54, 2, (__int64)&v57, 0);
    if ( !unk_4D04700 )
      return v10;
LABEL_10:
    if ( *(_BYTE *)(v10 + 16) > 0x17u )
    {
      v11 = sub_15F24E0(v10);
      sub_15F2440(v10, v11 | 1u);
    }
    return v10;
  }
  if ( (unsigned __int8)sub_127B3A0(a4) )
  {
    v22 = *(_BYTE *)(a2 + 16) <= 0x10u;
    v23 = (__int64 *)a1[1];
    v56 = 1;
    v54[0] = "div";
    v55 = 3;
    if ( v22 && *(_BYTE *)(a3 + 16) <= 0x10u )
      return sub_15A2C90(a2, a3, 0);
    LOWORD(v59) = 257;
    v26 = 18;
    v25 = a3;
  }
  else
  {
    v22 = *(_BYTE *)(a2 + 16) <= 0x10u;
    v23 = (__int64 *)a1[1];
    v56 = 1;
    v54[0] = "div";
    v55 = 3;
    if ( v22 && *(_BYTE *)(a3 + 16) <= 0x10u )
      return sub_15A2C70(a2, a3, 0);
    v25 = a3;
    LOWORD(v59) = 257;
    v26 = 17;
  }
  v27 = sub_15FB440(v26, a2, v25, &v57, 0);
  v28 = v23[1];
  v10 = v27;
  if ( v28 )
  {
    v29 = (__int64 *)v23[2];
    sub_157E9D0(v28 + 40, v27);
    v30 = *(_QWORD *)(v10 + 24);
    v31 = *v29;
    *(_QWORD *)(v10 + 32) = v29;
    v31 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v10 + 24) = v31 | v30 & 7;
    *(_QWORD *)(v31 + 8) = v10 + 24;
    *v29 = *v29 & 7 | (v10 + 24);
  }
  sub_164B780(v10, v54);
  v32 = *v23;
  if ( *v23 )
  {
    v53 = *v23;
    sub_1623A60(&v53, v32, 2);
    if ( *(_QWORD *)(v10 + 48) )
      sub_161E7C0(v10 + 48);
    v33 = v53;
    *(_QWORD *)(v10 + 48) = v53;
    if ( v33 )
      sub_1623210(&v53, v33, v10 + 48);
  }
  return v10;
}
