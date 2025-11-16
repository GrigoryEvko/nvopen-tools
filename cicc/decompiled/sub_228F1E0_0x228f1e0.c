// Function: sub_228F1E0
// Address: 0x228f1e0
//
__int64 __fastcall sub_228F1E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char *a5,
        int a6,
        __int64 a7,
        __int64 a8,
        __int64 *a9)
{
  __int64 v10; // r13
  _QWORD *v12; // r12
  unsigned int v13; // eax
  __int64 v14; // r9
  __int64 v15; // r9
  unsigned int v16; // r15d
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // r14
  _BYTE *v30; // rdx
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 *v33; // rax
  __int64 *v34; // rdi
  unsigned __int64 v35; // rax
  unsigned int v36; // eax
  __int64 v37; // [rsp+0h] [rbp-B0h]
  __int64 v38; // [rsp+8h] [rbp-A8h]
  __int64 *v39; // [rsp+10h] [rbp-A0h]
  _QWORD *v40; // [rsp+10h] [rbp-A0h]
  unsigned int v41; // [rsp+18h] [rbp-98h]
  __int64 *v42; // [rsp+18h] [rbp-98h]
  __int64 *v43; // [rsp+18h] [rbp-98h]
  __int64 v44; // [rsp+18h] [rbp-98h]
  __int64 *v45; // [rsp+18h] [rbp-98h]
  __int64 v46[2]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v47[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v48; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+48h] [rbp-68h]
  unsigned __int64 v50; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v51; // [rsp+58h] [rbp-58h]
  unsigned __int64 v52; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v53; // [rsp+68h] [rbp-48h]
  unsigned __int64 v54; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v55; // [rsp+78h] [rbp-38h]

  v10 = a2;
  v41 = a6 - 1;
  *(_BYTE *)(a7 + 43) = 0;
  v12 = sub_DCC810(*(__int64 **)(a1 + 8), a4, a3, 0, 0);
  sub_228CE50(a8, v10, v10, (__int64)v12, (__int64)a5);
  LOBYTE(v13) = sub_D968A0((__int64)v12);
  if ( (_BYTE)v13 )
  {
    v14 = 16LL * v41;
    *(_BYTE *)(v14 + *(_QWORD *)(a7 + 48)) &= ~1u;
    *(_BYTE *)(v14 + *(_QWORD *)(a7 + 48)) &= ~4u;
    v15 = *(_QWORD *)(a7 + 48) + v14;
    if ( (*(_BYTE *)v15 & 7) != 0 )
    {
      *(_QWORD *)(v15 + 8) = v12;
      return 0;
    }
    return 1;
  }
  v16 = v13;
  if ( !*(_WORD *)(a2 + 24) )
  {
    *(_BYTE *)(*(_QWORD *)(a7 + 48) + 16LL * v41) |= 0x40u;
    v37 = 16LL * v41;
    if ( (unsigned __int8)sub_DBEC00(*(_QWORD *)(a1 + 8), a2) )
    {
      v33 = sub_DCAF50(*(__int64 **)(a1 + 8), a2, 0);
      v34 = *(__int64 **)(a1 + 8);
      v10 = (__int64)v33;
      if ( *((_WORD *)v33 + 12) )
        v10 = 0;
      v12 = sub_DCAF50(v34, (__int64)v12, 0);
    }
    v42 = *(__int64 **)(a1 + 8);
    v18 = sub_D95540((__int64)v12);
    v19 = sub_DA2C50((__int64)v42, v18, 2, 0);
    v38 = (__int64)v42;
    v39 = sub_DCA690(v42, (__int64)v19, v10, 0, 0);
    v43 = *(__int64 **)(a1 + 8);
    v20 = sub_D95540((__int64)v12);
    v21 = sub_DA2C50((__int64)v43, v20, 0, 0);
    v23 = sub_DCDFA0(v43, (__int64)v21, (__int64)v12, v22);
    *a9 = sub_DCB270(v38, v23, (__int64)v39);
    if ( !*((_WORD *)v12 + 12) )
    {
      v16 = sub_DBEC00(*(_QWORD *)(a1 + 8), (__int64)v12);
      if ( !(_BYTE)v16 )
      {
        v24 = sub_D95540((__int64)v12);
        v25 = sub_228E360(a1, a5, v24);
        v26 = (__int64)v25;
        if ( !v25 )
        {
LABEL_18:
          sub_9865C0((__int64)v46, v12[4] + 24LL);
          sub_9865C0((__int64)v47, *(_QWORD *)(v10 + 32) + 24LL);
          sub_9865C0((__int64)&v48, (__int64)v46);
          sub_9865C0((__int64)&v50, (__int64)v46);
          sub_C4C400((__int64)v46, (__int64)v47, (__int64)&v48, (__int64)&v50);
          if ( sub_D94970((__int64)&v50, 0) )
          {
            v53 = v49;
            if ( v49 > 0x40 )
            {
              sub_C43690((__int64)&v52, 2, 1);
            }
            else
            {
              v35 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v49;
              if ( !v49 )
                LOBYTE(v35) = 0;
              v52 = v35 & 2;
            }
            sub_C4B8A0((__int64)&v54, (__int64)&v48, (__int64)&v52);
            if ( v51 > 0x40 && v50 )
              j_j___libc_free_0_0(v50);
            v50 = v54;
            v36 = v55;
            v55 = 0;
            v51 = v36;
            sub_969240((__int64 *)&v54);
            if ( !sub_D94970((__int64)&v50, 0) )
              *(_BYTE *)(*(_QWORD *)(a7 + 48) + v37) &= ~2u;
            sub_969240((__int64 *)&v52);
          }
          else
          {
            v16 = 1;
          }
          sub_969240((__int64 *)&v50);
          sub_969240(&v48);
          sub_969240(v47);
          sub_969240(v46);
          return v16;
        }
        v44 = *(_QWORD *)(a1 + 8);
        v27 = sub_D95540((__int64)v25);
        v40 = sub_DA2C50(v44, v27, 2, 0);
        v45 = *(__int64 **)(a1 + 8);
        v28 = sub_DCA690(v45, v10, v26, 0, 0);
        v29 = sub_DCA690(v45, (__int64)v28, (__int64)v40, 0, 0);
        if ( !sub_228DFC0(a1, 0x26u, (__int64)v12, (__int64)v29) )
        {
          if ( sub_228DFC0(a1, 0x20u, (__int64)v12, (__int64)v29) )
          {
            *(_BYTE *)(v37 + *(_QWORD *)(a7 + 48)) &= ~1u;
            *(_BYTE *)(v37 + *(_QWORD *)(a7 + 48)) &= ~4u;
            v30 = (_BYTE *)(*(_QWORD *)(a7 + 48) + v37);
            if ( (*v30 & 7) != 0 )
            {
              *v30 &= ~0x40u;
              v31 = *(_QWORD *)(a1 + 8);
              v32 = sub_D95540((__int64)v12);
              *(_QWORD *)(*(_QWORD *)(a7 + 48) + v37 + 8) = sub_DA2C50(v31, v32, 0, 0);
              return v16;
            }
            return 1;
          }
          goto LABEL_18;
        }
      }
      return 1;
    }
  }
  return v16;
}
