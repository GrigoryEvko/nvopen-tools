// Function: sub_3305B40
// Address: 0x3305b40
//
__int64 __fastcall sub_3305B40(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int128 a7)
{
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // r14
  int v15; // r15d
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // r9d
  __int64 v22; // r14
  __int128 *v23; // rbx
  __int64 v24; // rsi
  __int64 v25; // r15
  int v26; // r13d
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // r13
  int v34; // edx
  int v35; // r14d
  __int64 v36; // rax
  int v37; // edx
  __int128 v38; // [rsp-30h] [rbp-B0h]
  __int128 v39; // [rsp-10h] [rbp-90h]
  __int64 v40; // [rsp+0h] [rbp-80h]
  __int64 v41; // [rsp+8h] [rbp-78h]
  __int64 v43; // [rsp+18h] [rbp-68h]
  __int64 v44; // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+20h] [rbp-60h] BYREF
  int v46; // [rsp+28h] [rbp-58h]
  __int64 v47; // [rsp+30h] [rbp-50h] BYREF
  int v48; // [rsp+38h] [rbp-48h]
  __int64 v49; // [rsp+40h] [rbp-40h]
  int v50; // [rsp+48h] [rbp-38h]

  if ( (unsigned __int8)sub_33DFCF0(a2, a3, 0) && (v28 = sub_326C2C0(a7, *((__int64 *)&a7 + 1), *a1, 1), v29 = v27, v28) )
  {
    v30 = *(_QWORD *)(a6 + 80);
    v45 = v30;
    if ( v30 )
    {
      v41 = v27;
      v40 = v28;
      sub_B96E90((__int64)&v45, v30, 1);
      v28 = v40;
      v29 = v41;
    }
    v31 = *a1;
    v32 = *(_QWORD *)(a6 + 48);
    v46 = *(_DWORD *)(a6 + 72);
    *((_QWORD *)&v39 + 1) = v29;
    *(_QWORD *)&v39 = v28;
    *((_QWORD *)&v38 + 1) = a5;
    *(_QWORD *)&v38 = a4;
    v33 = sub_3412970(
            v31,
            73,
            (unsigned int)&v45,
            v32,
            *(_DWORD *)(a6 + 68),
            v29,
            v38,
            *(_OWORD *)*(_QWORD *)(a2 + 40),
            v39);
    v35 = v34;
    v36 = sub_3407510(
            *a1,
            &v45,
            v33,
            1,
            *(unsigned __int16 *)(*(_QWORD *)(v33 + 48) + 16LL),
            *(_QWORD *)(*(_QWORD *)(v33 + 48) + 24LL));
    v50 = v37;
    v47 = v33;
    v48 = v35;
    v49 = v36;
    result = sub_32EB790((__int64)a1, a6, &v47, 2, 1);
    if ( v45 )
    {
      v44 = result;
      sub_B91220((__int64)&v45, v45);
      return v44;
    }
  }
  else
  {
    v12 = *(_DWORD *)(a2 + 24);
    if ( v12 != 56 && (v12 != 77 || (_DWORD)a3 || (_QWORD)a7 == a2 && DWORD2(a7) == 1)
      || !(unsigned __int8)sub_33CF170(a4, a5)
      || (unsigned __int8)sub_33CF8A0(a6, 1, v17, v18, v19, v20) )
    {
      v14 = sub_32719C0((_DWORD *)a1[1], a4, a5, 0);
      v15 = v13;
      if ( !v14 )
        return 0;
      result = sub_32C2770((__int64)a1, *a1, a2, a3, v14, v13, a7, SDWORD2(a7), a6);
      if ( !result )
      {
        result = sub_32C2770((__int64)a1, *a1, a2, a3, a7, *((__int64 *)&a7 + 1), v14, v15, a6);
        if ( !result )
          return 0;
      }
    }
    else
    {
      v22 = *a1;
      v23 = *(__int128 **)(a2 + 40);
      v24 = *(_QWORD *)(a6 + 80);
      v25 = *(_QWORD *)(a6 + 48);
      v26 = *(_DWORD *)(a6 + 68);
      v47 = v24;
      if ( v24 )
        sub_B96E90((__int64)&v47, v24, 1);
      v48 = *(_DWORD *)(a6 + 72);
      result = sub_3412970(v22, 72, (unsigned int)&v47, v25, v26, v21, *v23, *(__int128 *)((char *)v23 + 40), a7);
      if ( v47 )
      {
        v43 = result;
        sub_B91220((__int64)&v47, v47);
        return v43;
      }
    }
  }
  return result;
}
