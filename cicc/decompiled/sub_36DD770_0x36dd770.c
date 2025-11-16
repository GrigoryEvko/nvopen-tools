// Function: sub_36DD770
// Address: 0x36dd770
//
void __fastcall sub_36DD770(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 *v9; // rax
  unsigned int v10; // r10d
  unsigned int v11; // r13d
  __int64 v12; // rsi
  __int64 v13; // r14
  __int64 v14; // r15
  int v15; // eax
  __int64 v16; // rdi
  int v17; // esi
  __int64 v18; // r13
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  int v22; // esi
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdi
  __int64 v28; // r14
  __int16 *v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // r13
  unsigned __int16 v32; // r8
  __int64 v33; // r15
  unsigned __int8 *v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 v38; // r13
  _DWORD *v39; // rax
  unsigned int v40; // r10d
  __int128 v41; // rax
  __int128 v42; // rax
  __int64 v43; // r9
  __int64 v44; // rax
  __int128 v45; // [rsp-30h] [rbp-90h]
  __int128 v46; // [rsp-10h] [rbp-70h]
  __int128 v47; // [rsp-10h] [rbp-70h]
  __int64 v48; // [rsp-10h] [rbp-70h]
  unsigned int v49; // [rsp+8h] [rbp-58h]
  unsigned __int16 v50; // [rsp+8h] [rbp-58h]
  unsigned int v51; // [rsp+8h] [rbp-58h]
  __int64 v52; // [rsp+10h] [rbp-50h] BYREF
  int v53; // [rsp+18h] [rbp-48h]
  __int64 v54; // [rsp+20h] [rbp-40h] BYREF
  int v55; // [rsp+28h] [rbp-38h]

  v9 = *(__int64 **)(a2 + 40);
  v10 = *(_DWORD *)(a2 + 96);
  v11 = *(_DWORD *)(a2 + 100);
  v12 = *(_QWORD *)(a2 + 80);
  v13 = *v9;
  v14 = v9[1];
  v52 = v12;
  if ( v12 )
  {
    v49 = v10;
    sub_B96E90((__int64)&v52, v12, 1);
    v10 = v49;
  }
  v15 = *(_DWORD *)(a2 + 72);
  v53 = v15;
  if ( !v11 )
  {
    v16 = *(_QWORD *)(a1 + 952);
    if ( *(_BYTE *)(v16 + 1264) )
    {
      v51 = v10;
      v39 = sub_AE2980(v16 + 16, v10);
      v40 = v51;
      if ( v39[1] == 32 )
      {
        *(_QWORD *)&v42 = sub_3400BD0(*(_QWORD *)(a1 + 64), 0, (__int64)&v52, 7, 0, 1u, a3, 0);
        *((_QWORD *)&v45 + 1) = v14;
        *(_QWORD *)&v45 = v13;
        v44 = sub_33F77A0(*(_QWORD **)(a1 + 64), 1216, (__int64)&v52, 8u, 0, v43, v45, v42);
        v40 = v51;
        v13 = v44;
        v14 &= 0xFFFFFFFF00000000LL;
      }
      if ( v40 == 3 )
      {
        v17 = 7001 - (*(_BYTE *)(*(_QWORD *)(a1 + 952) + 1264LL) == 0);
        goto LABEL_10;
      }
      if ( v40 > 3 )
      {
        if ( v40 == 4 )
        {
          v17 = 6995 - (*(_BYTE *)(*(_QWORD *)(a1 + 952) + 1264LL) == 0);
          goto LABEL_10;
        }
        if ( v40 == 5 )
        {
          v17 = 6999 - (*(_BYTE *)(*(_QWORD *)(a1 + 952) + 1264LL) == 0);
          goto LABEL_10;
        }
      }
      else if ( v40 == 1 )
      {
        v17 = 6997 - (*(_BYTE *)(*(_QWORD *)(a1 + 952) + 1264LL) == 0);
        goto LABEL_10;
      }
    }
    else
    {
      if ( v10 == 4 )
      {
        v17 = 6994;
        goto LABEL_10;
      }
      if ( v10 > 4 )
      {
        if ( v10 == 5 )
        {
          v17 = 6998;
          goto LABEL_10;
        }
      }
      else
      {
        if ( v10 == 1 )
        {
          v17 = 6996;
          goto LABEL_10;
        }
        if ( v10 == 3 )
        {
          v17 = 7000;
LABEL_10:
          *((_QWORD *)&v46 + 1) = v14;
          *(_QWORD *)&v46 = v13;
          v18 = sub_33F7740(
                  *(_QWORD **)(a1 + 64),
                  v17,
                  (__int64)&v52,
                  **(unsigned __int16 **)(a2 + 48),
                  *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                  a7,
                  v46);
          sub_34158F0(*(_QWORD *)(a1 + 64), a2, v18, v19, v20, v21);
          sub_3421DB0(v18);
          sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
          goto LABEL_11;
        }
      }
    }
LABEL_50:
    sub_C64ED0("Bad address space in addrspacecast", 1u);
  }
  if ( v10 )
  {
    v29 = *(__int16 **)(a2 + 48);
    v30 = *(_QWORD *)(a2 + 80);
    v31 = *(_QWORD *)(a1 + 64);
    v32 = *v29;
    v33 = *((_QWORD *)v29 + 1);
    v54 = v30;
    if ( v30 )
    {
      v50 = v32;
      sub_B96E90((__int64)&v54, v30, 1);
      v15 = *(_DWORD *)(a2 + 72);
      v32 = v50;
    }
    v55 = v15;
    v34 = sub_3400BD0(v31, 0, (__int64)&v54, v32, v33, 1u, a3, 0);
    v37 = v48;
    v38 = (__int64)v34;
    if ( v54 )
      sub_B91220((__int64)&v54, v54);
    sub_34158F0(*(_QWORD *)(a1 + 64), a2, v38, v37, v35, v36);
    sub_3421DB0(v38);
    sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  }
  else
  {
    if ( v11 == 4 )
    {
      v22 = 7002 - ((*(_BYTE *)(*(_QWORD *)(a1 + 952) + 1264LL) == 0) - 1);
    }
    else if ( v11 > 4 )
    {
      if ( v11 == 5 )
      {
        v22 = 7006 - ((*(_BYTE *)(*(_QWORD *)(a1 + 952) + 1264LL) == 0) - 1);
      }
      else
      {
        if ( v11 != 101 )
          goto LABEL_50;
        v22 = *(_BYTE *)(*(_QWORD *)(a1 + 952) + 1264LL) == 0 ? 1604 : 1606;
      }
    }
    else if ( v11 == 1 )
    {
      v22 = 7004 - ((*(_BYTE *)(*(_QWORD *)(a1 + 952) + 1264LL) == 0) - 1);
    }
    else
    {
      if ( v11 != 3 )
        goto LABEL_50;
      v22 = 7008 - ((*(_BYTE *)(*(_QWORD *)(a1 + 952) + 1264LL) == 0) - 1);
    }
    *((_QWORD *)&v47 + 1) = v14;
    *(_QWORD *)&v47 = v13;
    v23 = sub_33F7740(
            *(_QWORD **)(a1 + 64),
            v22,
            (__int64)&v52,
            **(unsigned __int16 **)(a2 + 48),
            *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
            a7,
            v47);
    v27 = *(_QWORD *)(a1 + 952);
    v28 = v23;
    if ( *(_BYTE *)(v27 + 1264) && sub_AE2980(v27 + 16, v11)[1] == 32 )
    {
      *(_QWORD *)&v41 = sub_3400BD0(*(_QWORD *)(a1 + 64), 0, (__int64)&v52, 7, 0, 1u, a3, 0);
      v28 = sub_33F77A0(
              *(_QWORD **)(a1 + 64),
              1205,
              (__int64)&v52,
              7u,
              0,
              *(_QWORD *)(a1 + 64),
              (unsigned __int64)v28,
              v41);
    }
    sub_34158F0(*(_QWORD *)(a1 + 64), a2, v28, v24, v25, v26);
    sub_3421DB0(v28);
    sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  }
LABEL_11:
  if ( v52 )
    sub_B91220((__int64)&v52, v52);
}
