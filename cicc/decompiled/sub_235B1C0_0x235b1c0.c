// Function: sub_235B1C0
// Address: 0x235b1c0
//
__int64 __fastcall sub_235B1C0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // bl
  __int64 v8; // r15
  __int16 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  char *v14; // rdi
  char v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  char *v20; // rax
  char *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  char *v29; // rdx
  char *v30; // rdx
  char v32; // [rsp+0h] [rbp-150h]
  __int64 v33; // [rsp+0h] [rbp-150h]
  char v34; // [rsp+Ch] [rbp-144h]
  __int64 v35; // [rsp+18h] [rbp-138h] BYREF
  char *v36; // [rsp+20h] [rbp-130h] BYREF
  __int64 v37; // [rsp+28h] [rbp-128h]
  char v38; // [rsp+30h] [rbp-120h] BYREF
  int v39; // [rsp+60h] [rbp-F0h]
  __int64 v40; // [rsp+68h] [rbp-E8h]
  __int64 v41; // [rsp+70h] [rbp-E0h]
  __int64 v42; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v43; // [rsp+80h] [rbp-D0h] BYREF
  char *v44; // [rsp+88h] [rbp-C8h]
  char *v45; // [rsp+90h] [rbp-C0h]
  char *v46; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+A8h] [rbp-A8h]
  char v48; // [rsp+B0h] [rbp-A0h] BYREF
  int v49; // [rsp+E0h] [rbp-70h]
  __int64 v50; // [rsp+E8h] [rbp-68h]
  __int64 v51; // [rsp+F0h] [rbp-60h]
  __int64 v52; // [rsp+F8h] [rbp-58h]
  unsigned __int64 v53; // [rsp+100h] [rbp-50h]
  char *v54; // [rsp+108h] [rbp-48h]
  char *v55; // [rsp+110h] [rbp-40h]

  v7 = a5;
  v36 = &v38;
  v34 = a3;
  v32 = a4;
  v37 = 0x600000000LL;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  sub_2332320((__int64)&v36, 1, a3, a4, a5, a6);
  v8 = *a2;
  v9 = *((_WORD *)a2 + 4);
  v10 = sub_22077B0(0x18u);
  v14 = (char *)v10;
  if ( v10 )
  {
    *(_QWORD *)(v10 + 8) = v8;
    *(_WORD *)(v10 + 16) = v9;
    *(_QWORD *)v10 = &unk_4A140B8;
  }
  v46 = (char *)v10;
  if ( v44 == v45 )
  {
    sub_235B010(&v43, v44, &v46);
    v14 = v46;
  }
  else
  {
    if ( v44 )
    {
      *(_QWORD *)v44 = v10;
      v44 += 8;
      goto LABEL_6;
    }
    v44 = (char *)8;
  }
  if ( v14 )
    (*(void (__fastcall **)(char *))(*(_QWORD *)v14 + 8LL))(v14);
LABEL_6:
  v15 = v32;
  v46 = &v48;
  v47 = 0x600000000LL;
  if ( (_DWORD)v37 )
    sub_2303E40((__int64)&v46, &v36, v11, (unsigned int)v37, v12, v13);
  v49 = v39;
  v16 = v40;
  v40 = 0;
  v50 = v16;
  v17 = v41;
  v41 = 0;
  v51 = v17;
  v18 = v42;
  v42 = 0;
  v52 = v18;
  v19 = v43;
  v43 = 0;
  v53 = v19;
  v20 = v44;
  v44 = 0;
  v54 = v20;
  v21 = v45;
  v45 = 0;
  v55 = v21;
  v22 = sub_22077B0(0x80u);
  if ( v22 )
  {
    *(_QWORD *)(v22 + 16) = 0x600000000LL;
    *(_QWORD *)v22 = &unk_4A0B4E8;
    *(_QWORD *)(v22 + 8) = v22 + 24;
    if ( (_DWORD)v47 )
    {
      v33 = v22;
      sub_2303E40(v22 + 8, &v46, (unsigned int)v47, 0x600000000LL, v23, v24);
      v22 = v33;
    }
    *(_DWORD *)(v22 + 72) = v49;
    v25 = v50;
    v50 = 0;
    *(_QWORD *)(v22 + 80) = v25;
    v26 = v51;
    v51 = 0;
    *(_QWORD *)(v22 + 88) = v26;
    v27 = v52;
    v52 = 0;
    *(_QWORD *)(v22 + 96) = v27;
    v28 = v53;
    v53 = 0;
    *(_QWORD *)(v22 + 104) = v28;
    v29 = v54;
    v54 = 0;
    *(_QWORD *)(v22 + 112) = v29;
    v30 = v55;
    v55 = 0;
    *(_QWORD *)(v22 + 120) = v30;
  }
  v35 = v22;
  sub_2354930(a1, &v35, v34, v15, v7, 1);
  if ( v35 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v35 + 8LL))(v35);
  sub_2337B30((unsigned __int64 *)&v46);
  sub_2337B30((unsigned __int64 *)&v36);
  return a1;
}
