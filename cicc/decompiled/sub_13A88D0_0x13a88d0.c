// Function: sub_13A88D0
// Address: 0x13a88d0
//
__int64 __fastcall sub_13A88D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8,
        _QWORD *a9)
{
  __int64 v10; // r13
  __int64 v12; // r12
  unsigned int v13; // eax
  __int64 v14; // r9
  __int64 v15; // r9
  unsigned int v16; // r15d
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r14
  _BYTE *v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // [rsp+0h] [rbp-B0h]
  __int64 v36; // [rsp+8h] [rbp-A8h]
  __int64 v37; // [rsp+10h] [rbp-A0h]
  __int64 v38; // [rsp+10h] [rbp-A0h]
  unsigned int v39; // [rsp+18h] [rbp-98h]
  __int64 v40; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+18h] [rbp-98h]
  __int64 v42; // [rsp+18h] [rbp-98h]
  __int64 v43; // [rsp+18h] [rbp-98h]
  __int64 v44[2]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v45[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v46; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v47; // [rsp+48h] [rbp-68h]
  __int64 v48[2]; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v49; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v50; // [rsp+68h] [rbp-48h]
  __int64 v51[8]; // [rsp+70h] [rbp-40h] BYREF

  v10 = a2;
  v39 = a6 - 1;
  *(_BYTE *)(a7 + 43) = 0;
  v12 = sub_14806B0(*(_QWORD *)(a1 + 8), a4, a3, 0, 0);
  sub_13A62E0(a8, v10, v10, v12, a5);
  v13 = sub_14560B0(v12);
  if ( (_BYTE)v13 )
  {
    v14 = 16LL * v39;
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
    *(_BYTE *)(*(_QWORD *)(a7 + 48) + 16LL * v39) |= 0x40u;
    v35 = 16LL * v39;
    if ( (unsigned __int8)sub_1477B50(*(_QWORD *)(a1 + 8), a2) )
    {
      v33 = sub_1480620(*(_QWORD *)(a1 + 8), a2, 0);
      v34 = *(_QWORD *)(a1 + 8);
      v10 = v33;
      if ( *(_WORD *)(v33 + 24) )
        v10 = 0;
      v12 = sub_1480620(v34, v12, 0);
    }
    v40 = *(_QWORD *)(a1 + 8);
    v18 = sub_1456040(v12);
    v19 = sub_145CF80(v40, v18, 2, 0);
    v36 = v40;
    v37 = sub_13A5B60(v40, v19, v10, 0, 0);
    v41 = *(_QWORD *)(a1 + 8);
    v20 = sub_1456040(v12);
    v21 = sub_145CF80(v41, v20, 0, 0);
    v22 = sub_147A9C0(v41, v21, v12);
    *a9 = sub_1483CF0(v36, v22, v37);
    if ( !*(_WORD *)(v12 + 24) )
    {
      v16 = sub_1477B50(*(_QWORD *)(a1 + 8), v12);
      if ( !(_BYTE)v16 )
      {
        v23 = sub_1456040(v12);
        v24 = sub_13A7AF0(a1, a5, v23);
        v25 = v24;
        if ( !v24 )
        {
LABEL_18:
          sub_13A38D0((__int64)v44, *(_QWORD *)(v12 + 32) + 24LL);
          sub_13A38D0((__int64)v45, *(_QWORD *)(v10 + 32) + 24LL);
          sub_13A38D0((__int64)&v46, (__int64)v44);
          sub_13A38D0((__int64)v48, (__int64)v44);
          sub_16AE5C0(v44, v45, &v46, v48);
          if ( sub_13A38F0((__int64)v48, 0) )
          {
            v50 = v47;
            if ( v47 > 0x40 )
              sub_16A4EF0(&v49, 2, 1);
            else
              v49 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v47) & 2;
            sub_16AB4D0(v51, &v46, &v49);
            sub_13A3610(v48, v51);
            sub_135E100(v51);
            if ( !sub_13A38F0((__int64)v48, 0) )
              *(_BYTE *)(*(_QWORD *)(a7 + 48) + v35) &= ~2u;
            sub_135E100((__int64 *)&v49);
          }
          else
          {
            v16 = 1;
          }
          sub_135E100(v48);
          sub_135E100(&v46);
          sub_135E100(v45);
          sub_135E100(v44);
          return v16;
        }
        v42 = *(_QWORD *)(a1 + 8);
        v26 = sub_1456040(v24);
        v38 = sub_145CF80(v42, v26, 2, 0);
        v43 = *(_QWORD *)(a1 + 8);
        v27 = sub_13A5B60(v43, v10, v25, 0, 0);
        v28 = sub_13A5B60(v43, v27, v38, 0, 0);
        if ( !(unsigned __int8)sub_13A7760(a1, 38, v12, v28) )
        {
          if ( (unsigned __int8)sub_13A7760(a1, 32, v12, v28) )
          {
            *(_BYTE *)(v35 + *(_QWORD *)(a7 + 48)) &= ~1u;
            *(_BYTE *)(v35 + *(_QWORD *)(a7 + 48)) &= ~4u;
            v29 = (_BYTE *)(*(_QWORD *)(a7 + 48) + v35);
            if ( (*v29 & 7) != 0 )
            {
              *v29 &= ~0x40u;
              v30 = *(_QWORD *)(a1 + 8);
              v31 = sub_1456040(v12);
              v32 = *(_QWORD *)(a7 + 48) + v35;
              *(_QWORD *)(v32 + 8) = sub_145CF80(v30, v31, 0, 0);
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
