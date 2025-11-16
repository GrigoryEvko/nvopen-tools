// Function: sub_17E8630
// Address: 0x17e8630
//
void __fastcall sub_17E8630(__int64 a1, unsigned __int8 *a2)
{
  int v3; // eax
  _BYTE *v5; // rsi
  _QWORD *v6; // rax
  unsigned __int8 *v7; // rsi
  __int64 v8; // rax
  __int64 **v9; // r15
  __int64 **v10; // rax
  __int64 ***v11; // rdi
  __int64 v12; // rbx
  unsigned __int8 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rax
  unsigned __int8 *v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int8 *v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rsi
  int v25; // edi
  __int64 *v26; // r15
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rsi
  __int64 v30; // rdx
  unsigned __int8 *v31; // rsi
  __int64 v32; // [rsp-140h] [rbp-140h]
  unsigned int v33; // [rsp-138h] [rbp-138h]
  __int64 *v34; // [rsp-130h] [rbp-130h]
  unsigned __int8 *v35; // [rsp-120h] [rbp-120h] BYREF
  __int64 v36; // [rsp-118h] [rbp-118h] BYREF
  __int16 v37; // [rsp-108h] [rbp-108h]
  __int64 v38; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v39; // [rsp-E8h] [rbp-E8h]
  __int64 v40; // [rsp-D8h] [rbp-D8h] BYREF
  __int16 v41; // [rsp-C8h] [rbp-C8h]
  unsigned __int8 *v42[6]; // [rsp-B8h] [rbp-B8h] BYREF
  unsigned __int8 *v43; // [rsp-88h] [rbp-88h] BYREF
  __int64 v44; // [rsp-80h] [rbp-80h]
  unsigned __int8 *v45; // [rsp-78h] [rbp-78h]
  _QWORD *v46; // [rsp-70h] [rbp-70h]
  __int64 v47; // [rsp-68h] [rbp-68h]
  int v48; // [rsp-60h] [rbp-60h]
  __int64 v49; // [rsp-58h] [rbp-58h]
  __int64 v50; // [rsp-50h] [rbp-50h]

  if ( !byte_4FA4FA0 || *(_BYTE *)(*(_QWORD *)&a2[24 * (2LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))] + 16LL) == 13 )
    return;
  v3 = *(_DWORD *)(a1 + 12);
  if ( v3 != 1 )
  {
    if ( v3 == 2 )
    {
      v43 = a2;
      v5 = *(_BYTE **)(a1 + 56);
      if ( v5 == *(_BYTE **)(a1 + 64) )
      {
        sub_17C2330(a1 + 48, v5, &v43);
      }
      else
      {
        if ( v5 )
        {
          *(_QWORD *)v5 = a2;
          v5 = *(_BYTE **)(a1 + 56);
        }
        *(_QWORD *)(a1 + 56) = v5 + 8;
      }
    }
    else
    {
      ++*(_DWORD *)(a1 + 8);
    }
    return;
  }
  v34 = *(__int64 **)(*(_QWORD *)a1 + 40LL);
  v6 = (_QWORD *)sub_16498A0((__int64)a2);
  v7 = (unsigned __int8 *)*((_QWORD *)a2 + 6);
  v43 = 0;
  v46 = v6;
  v8 = *((_QWORD *)a2 + 5);
  v47 = 0;
  v44 = v8;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v45 = a2 + 24;
  v42[0] = v7;
  if ( v7 )
  {
    sub_1623A60((__int64)v42, (__int64)v7, 2);
    if ( v43 )
      sub_161E7C0((__int64)&v43, (__int64)v43);
    v43 = v42[0];
    if ( v42[0] )
      sub_1623210((__int64)v42, v42[0], (__int64)&v43);
  }
  v9 = (__int64 **)sub_1643360(v46);
  v10 = (__int64 **)sub_16471D0(v46, 0);
  v11 = *(__int64 ****)(a1 + 24);
  v12 = *(_QWORD *)&a2[24 * (2LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))];
  v39 = 257;
  v13 = (unsigned __int8 *)sub_15A4510(v11, v10, 0);
  v14 = *(_QWORD *)(a1 + 32);
  v42[0] = v13;
  v15 = sub_1643360(v46);
  v37 = 257;
  v42[1] = (unsigned __int8 *)sub_159C470(v15, v14, 0);
  v32 = *(_QWORD *)v12;
  v33 = sub_16431D0(*(_QWORD *)v12);
  v16 = sub_16431D0((__int64)v9);
  if ( v33 < v16 )
  {
    if ( v9 != (__int64 **)v32 )
    {
      if ( *(_BYTE *)(v12 + 16) > 0x10u )
      {
        v23 = (__int64)v9;
        v24 = v12;
        v41 = 257;
        v25 = 37;
        goto LABEL_29;
      }
      v12 = sub_15A46C0(37, (__int64 ***)v12, v9, 0);
    }
  }
  else if ( v9 != (__int64 **)v32 && v33 != v16 )
  {
    if ( *(_BYTE *)(v12 + 16) <= 0x10u )
    {
      v12 = sub_15A46C0(36, (__int64 ***)v12, v9, 0);
      goto LABEL_22;
    }
    v23 = (__int64)v9;
    v24 = v12;
    v41 = 257;
    v25 = 36;
LABEL_29:
    v12 = sub_15FDBD0(v25, v24, v23, (__int64)&v40, 0);
    if ( v44 )
    {
      v26 = (__int64 *)v45;
      sub_157E9D0(v44 + 40, v12);
      v27 = *(_QWORD *)(v12 + 24);
      v28 = *v26;
      *(_QWORD *)(v12 + 32) = v26;
      v28 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v12 + 24) = v28 | v27 & 7;
      *(_QWORD *)(v28 + 8) = v12 + 24;
      *v26 = *v26 & 7 | (v12 + 24);
    }
    sub_164B780(v12, &v36);
    if ( v43 )
    {
      v35 = v43;
      sub_1623A60((__int64)&v35, (__int64)v43, 2);
      v29 = *(_QWORD *)(v12 + 48);
      v30 = v12 + 48;
      if ( v29 )
      {
        sub_161E7C0(v12 + 48, v29);
        v30 = v12 + 48;
      }
      v31 = v35;
      *(_QWORD *)(v12 + 48) = v35;
      if ( v31 )
        sub_1623210((__int64)&v35, v31, v30);
    }
  }
LABEL_22:
  v42[2] = (unsigned __int8 *)v12;
  v17 = sub_1643350(v46);
  v18 = (unsigned __int8 *)sub_159C470(v17, 1, 0);
  v19 = *(unsigned int *)(a1 + 16);
  v42[3] = v18;
  v20 = sub_1643350(v46);
  v42[4] = (unsigned __int8 *)sub_159C470(v20, v19, 0);
  v21 = sub_15E26F0(v34, 112, 0, 0);
  sub_17E28C0((__int64)&v43, *(_QWORD *)(v21 + 24), v21, (__int64 *)v42, 5, &v38, 0);
  v22 = v43;
  ++*(_DWORD *)(a1 + 16);
  if ( v22 )
    sub_161E7C0((__int64)&v43, (__int64)v22);
}
