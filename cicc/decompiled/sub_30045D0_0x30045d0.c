// Function: sub_30045D0
// Address: 0x30045d0
//
__int64 __fastcall sub_30045D0(_QWORD *a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rdx
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // rdi
  __int64 (*v8)(); // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned int v29; // r12d
  _QWORD *v31[5]; // [rsp+0h] [rbp-130h] BYREF
  __int64 v32; // [rsp+28h] [rbp-108h]
  __int64 v33; // [rsp+30h] [rbp-100h]
  __int64 v34; // [rsp+38h] [rbp-F8h]
  int v35; // [rsp+40h] [rbp-F0h]
  __int64 v36; // [rsp+48h] [rbp-E8h]
  __int64 v37; // [rsp+50h] [rbp-E0h]
  __int64 v38; // [rsp+58h] [rbp-D8h]
  __int64 v39; // [rsp+60h] [rbp-D0h]
  unsigned int v40; // [rsp+68h] [rbp-C8h]
  __int64 v41; // [rsp+70h] [rbp-C0h]
  char *v42; // [rsp+78h] [rbp-B8h]
  __int64 v43; // [rsp+80h] [rbp-B0h]
  int v44; // [rsp+88h] [rbp-A8h]
  char v45; // [rsp+8Ch] [rbp-A4h]
  char v46; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v47; // [rsp+D0h] [rbp-60h]
  __int64 v48; // [rsp+D8h] [rbp-58h]
  __int64 v49; // [rsp+E0h] [rbp-50h]
  unsigned int v50; // [rsp+E8h] [rbp-48h]
  __int64 v51; // [rsp+F0h] [rbp-40h]
  __int64 v52; // [rsp+F8h] [rbp-38h]
  __int64 v53; // [rsp+100h] [rbp-30h]
  unsigned int v54; // [rsp+108h] [rbp-28h]

  v3 = a2[2];
  v31[0] = a2;
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 128LL);
  v5 = 0;
  if ( v4 != sub_2DAC790 )
  {
    v5 = (_QWORD *)((__int64 (__fastcall *)(__int64))v4)(v3);
    v3 = a2[2];
  }
  v31[1] = v5;
  v6 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 200LL))(v3);
  v7 = a2[2];
  v31[2] = v6;
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 216LL);
  v9 = 0;
  if ( v8 != sub_2F391C0 )
    v9 = (_QWORD *)((__int64 (__fastcall *)(__int64))v8)(v7);
  v31[3] = v9;
  v10 = (_QWORD *)a2[4];
  v11 = a1[1];
  v32 = 0;
  v31[4] = v10;
  v12 = a2[1];
  v33 = 0;
  LODWORD(v12) = *(_DWORD *)(v12 + 648);
  v34 = 0;
  v36 = 0;
  v35 = v12;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = &v46;
  v43 = 8;
  v44 = 0;
  v45 = 1;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v13 = sub_B82360(v11, (__int64)&unk_501EB14);
  if ( v13 && (v14 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v13 + 104LL))(v13, &unk_501EB14)) != 0 )
    v15 = v14 + 200;
  else
    v15 = 0;
  v16 = a1[1];
  v32 = v15;
  v17 = sub_B82360(v16, (__int64)&unk_501EACC);
  if ( v17 && (v18 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v17 + 104LL))(v17, &unk_501EACC)) != 0 )
    v19 = v18 + 200;
  else
    v19 = 0;
  v20 = a1[1];
  v33 = v19;
  v21 = sub_B82360(v20, (__int64)&unk_4F86530);
  if ( v21 && (v22 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v21 + 104LL))(v21, &unk_4F86530)) != 0 )
    v23 = *(_QWORD *)(v22 + 176);
  else
    v23 = 0;
  v24 = *a2;
  v34 = v23;
  if ( (unsigned __int8)sub_BB98D0(a1, v24) )
    v35 = 0;
  v29 = sub_3000EC0(v31, v24, v25, v26, v27, v28);
  sub_C7D6A0(v52, 8LL * v54, 4);
  sub_C7D6A0(v48, 8LL * v50, 4);
  if ( !v45 )
    _libc_free((unsigned __int64)v42);
  sub_C7D6A0(v38, 16LL * v40, 8);
  return v29;
}
