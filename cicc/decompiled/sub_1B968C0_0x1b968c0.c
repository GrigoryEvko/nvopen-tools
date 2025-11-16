// Function: sub_1B968C0
// Address: 0x1b968c0
//
__int64 __fastcall sub_1B968C0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        unsigned int a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v11; // rdi
  double v13; // xmm0_8
  __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // r15d
  __int64 v23; // r8
  __int64 v24; // rdi
  __int64 v25; // r8
  __int64 *v26; // rbx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 *v33; // r13
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 *v38; // r14
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // [rsp+0h] [rbp-A0h]
  __int64 v42; // [rsp+0h] [rbp-A0h]
  __int64 v43; // [rsp+0h] [rbp-A0h]
  __int64 *v44; // [rsp+8h] [rbp-98h]
  __int64 v45; // [rsp+10h] [rbp-90h] BYREF
  __int16 v46; // [rsp+20h] [rbp-80h]
  __int64 v47[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v48; // [rsp+40h] [rbp-60h]
  __int64 v49[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v50; // [rsp+60h] [rbp-40h]

  v11 = *(_QWORD *)a2;
  if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)a2 + 8LL) - 1) > 5u )
  {
    v29 = sub_15A0680(v11, a3, 0);
    v48 = 259;
    v47[0] = (__int64)"induction";
    v46 = 257;
    if ( *(_BYTE *)(v29 + 16) > 0x10u || *(_BYTE *)(a4 + 16) > 0x10u )
    {
      v50 = 257;
      v36 = sub_15FB440(15, (__int64 *)v29, a4, (__int64)v49, 0);
      v37 = *(_QWORD *)(a1 + 104);
      v30 = v36;
      if ( v37 )
      {
        v38 = *(__int64 **)(a1 + 112);
        sub_157E9D0(v37 + 40, v36);
        v39 = *(_QWORD *)(v30 + 24);
        v40 = *v38;
        *(_QWORD *)(v30 + 32) = v38;
        v40 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v30 + 24) = v40 | v39 & 7;
        *(_QWORD *)(v40 + 8) = v30 + 24;
        *v38 = *v38 & 7 | (v30 + 24);
      }
      sub_164B780(v30, &v45);
      sub_12A86E0((__int64 *)(a1 + 96), v30);
    }
    else
    {
      v30 = sub_15A2C20((__int64 *)v29, a4, 0, 0, a6, a7, a8);
    }
    if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(v30 + 16) > 0x10u )
    {
      v50 = 257;
      v31 = sub_15FB440(11, (__int64 *)a2, v30, (__int64)v49, 0);
      v32 = *(_QWORD *)(a1 + 104);
      v18 = v31;
      if ( v32 )
      {
        v33 = *(__int64 **)(a1 + 112);
        sub_157E9D0(v32 + 40, v31);
        v34 = *(_QWORD *)(v18 + 24);
        v35 = *v33;
        *(_QWORD *)(v18 + 32) = v33;
        v35 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v18 + 24) = v35 | v34 & 7;
        *(_QWORD *)(v35 + 8) = v18 + 24;
        *v33 = *v33 & 7 | (v18 + 24);
      }
      sub_164B780(v18, v47);
      sub_12A86E0((__int64 *)(a1 + 96), v18);
    }
    else
    {
      return sub_15A2B30((__int64 *)a2, v30, 0, 0, a6, a7, a8);
    }
  }
  else
  {
    v13 = (double)a3;
    v14 = sub_15A10B0(v11, (double)a3);
    v48 = 257;
    v44 = (__int64 *)(a1 + 96);
    if ( *(_BYTE *)(v14 + 16) > 0x10u
      || *(_BYTE *)(a4 + 16) > 0x10u
      || (v15 = sub_15A2A30((__int64 *)0x10, (__int64 *)v14, a4, 0, 0, v13, a7, a8)) == 0 )
    {
      v50 = 257;
      v20 = sub_15FB440(16, (__int64 *)v14, a4, (__int64)v49, 0);
      v21 = *(_QWORD *)(a1 + 128);
      v22 = *(_DWORD *)(a1 + 136);
      v23 = v20;
      if ( v21 )
      {
        v41 = v20;
        sub_1625C10(v20, 3, v21);
        v23 = v41;
      }
      v42 = v23;
      sub_15F2440(v23, v22);
      v24 = *(_QWORD *)(a1 + 104);
      v25 = v42;
      if ( v24 )
      {
        v26 = *(__int64 **)(a1 + 112);
        sub_157E9D0(v24 + 40, v42);
        v25 = v42;
        v27 = *v26;
        v28 = *(_QWORD *)(v42 + 24);
        *(_QWORD *)(v42 + 32) = v26;
        v27 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v42 + 24) = v27 | v28 & 7;
        *(_QWORD *)(v27 + 8) = v42 + 24;
        *v26 = *v26 & 7 | (v42 + 24);
      }
      v43 = v25;
      sub_164B780(v25, v47);
      sub_12A86E0(v44, v43);
      v15 = v43;
    }
    v16 = sub_1B8ED40(v15);
    v50 = 257;
    v17 = sub_1904E90((__int64)v44, a5, a2, v16, v49, 0, v13, a7, a8);
    return sub_1B8ED40(v17);
  }
  return v18;
}
