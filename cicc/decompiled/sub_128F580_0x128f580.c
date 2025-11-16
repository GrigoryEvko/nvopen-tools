// Function: sub_128F580
// Address: 0x128f580
//
__int64 __fastcall sub_128F580(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v7; // r15
  __int64 v8; // r14
  char *v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  char *v13; // r14
  char v14; // al
  __int64 *v15; // rdi
  __int64 v16; // r15
  int v17; // eax
  __int64 *v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  int v24; // esi
  __int64 v25; // rdi
  __int64 *v26; // r14
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rax
  _QWORD *v33; // rax
  __int64 v34; // rax
  int v35; // esi
  __int64 v36; // rdi
  __int64 *v37; // r13
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // [rsp+0h] [rbp-A0h]
  __int64 v41; // [rsp+0h] [rbp-A0h]
  unsigned int v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+8h] [rbp-98h]
  unsigned int v44; // [rsp+10h] [rbp-90h]
  __int64 v45; // [rsp+10h] [rbp-90h]
  unsigned __int16 v46; // [rsp+18h] [rbp-88h]
  __int64 *v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+28h] [rbp-78h] BYREF
  char *v49; // [rsp+30h] [rbp-70h] BYREF
  char v50; // [rsp+40h] [rbp-60h]
  char v51; // [rsp+41h] [rbp-5Fh]
  _QWORD v52[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v53; // [rsp+60h] [rbp-40h]

  v7 = *(__int64 **)(a2 + 72);
  v44 = a3;
  v42 = a4;
  v8 = v7[2];
  v46 = a5;
  v9 = sub_128D0F0(a1, (__int64)v7, a3, a4, a5);
  v13 = sub_128D0F0(a1, v8, v10, v11, v12);
  v14 = *(_BYTE *)(*(_QWORD *)v9 + 8LL);
  if ( v14 == 16 )
    v14 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v9 + 16LL) + 8LL);
  if ( (unsigned __int8)(v14 - 1) > 5u )
  {
    if ( (unsigned __int8)sub_127B3A0(*v7) )
    {
      v19 = a1[1];
      v51 = 1;
      v50 = 3;
      v47 = v19;
      v49 = "cmp";
      if ( (unsigned __int8)v9[16] <= 0x10u && (unsigned __int8)v13[16] <= 0x10u )
      {
        v16 = sub_15A37B0(v42, v9, v13, 0);
        return sub_128B370((__int64 *)a1, (_QWORD *)v16, 0, *(_QWORD *)a2, (_DWORD *)(a2 + 36));
      }
      v53 = 257;
      v21 = sub_1648A60(56, 2);
      v16 = v21;
      if ( v21 )
      {
        v45 = v21;
        v22 = *(_QWORD **)v9;
        if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16 )
        {
          v40 = v22[4];
          v23 = sub_1643320(*v22);
          v24 = sub_16463B0(v23, v40);
        }
        else
        {
          v24 = sub_1643320(*v22);
        }
        sub_15FEC10(v16, v24, 51, v42, (_DWORD)v9, (_DWORD)v13, (__int64)v52, 0);
      }
      else
      {
        v45 = 0;
      }
      v25 = v47[1];
      if ( v25 )
      {
        v26 = (__int64 *)v47[2];
        sub_157E9D0(v25 + 40, v16);
        v27 = *(_QWORD *)(v16 + 24);
        v28 = *v26;
        *(_QWORD *)(v16 + 32) = v26;
        v28 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v16 + 24) = v28 | v27 & 7;
        *(_QWORD *)(v28 + 8) = v16 + 24;
        *v26 = *v26 & 7 | (v16 + 24);
      }
      v29 = v45;
    }
    else
    {
      v20 = a1[1];
      v51 = 1;
      v50 = 3;
      v47 = v20;
      v49 = "cmp";
      if ( (unsigned __int8)v9[16] <= 0x10u && (unsigned __int8)v13[16] <= 0x10u )
      {
        v16 = sub_15A37B0(v44, v9, v13, 0);
        return sub_128B370((__int64 *)a1, (_QWORD *)v16, 0, *(_QWORD *)a2, (_DWORD *)(a2 + 36));
      }
      v53 = 257;
      v32 = sub_1648A60(56, 2);
      v16 = v32;
      if ( v32 )
      {
        v43 = v32;
        v33 = *(_QWORD **)v9;
        if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16 )
        {
          v41 = v33[4];
          v34 = sub_1643320(*v33);
          v35 = sub_16463B0(v34, v41);
        }
        else
        {
          v35 = sub_1643320(*v33);
        }
        sub_15FEC10(v16, v35, 51, v44, (_DWORD)v9, (_DWORD)v13, (__int64)v52, 0);
      }
      else
      {
        v43 = 0;
      }
      v36 = v47[1];
      if ( v36 )
      {
        v37 = (__int64 *)v47[2];
        sub_157E9D0(v36 + 40, v16);
        v38 = *(_QWORD *)(v16 + 24);
        v39 = *v37;
        *(_QWORD *)(v16 + 32) = v37;
        v39 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v16 + 24) = v39 | v38 & 7;
        *(_QWORD *)(v39 + 8) = v16 + 24;
        *v37 = *v37 & 7 | (v16 + 24);
      }
      v29 = v43;
    }
    sub_164B780(v29, &v49);
    v30 = *v47;
    if ( *v47 )
    {
      v48 = *v47;
      sub_1623A60(&v48, v30, 2);
      if ( *(_QWORD *)(v16 + 48) )
        sub_161E7C0(v16 + 48);
      v31 = v48;
      *(_QWORD *)(v16 + 48) = v48;
      if ( v31 )
        sub_1623210(&v48, v31, v16 + 48);
    }
  }
  else
  {
    v15 = a1[1];
    v52[0] = "cmp";
    v53 = 259;
    v16 = sub_1289B20(v15, v46, v9, (__int64)v13, (__int64)v52, 0);
    if ( unk_4D04700 && *(_BYTE *)(v16 + 16) > 0x17u )
    {
      v17 = sub_15F24E0(v16);
      sub_15F2440(v16, v17 | 1u);
    }
  }
  return sub_128B370((__int64 *)a1, (_QWORD *)v16, 0, *(_QWORD *)a2, (_DWORD *)(a2 + 36));
}
