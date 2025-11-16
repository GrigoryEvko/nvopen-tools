// Function: sub_235D930
// Address: 0x235d930
//
__int64 __fastcall sub_235D930(__int64 a1, int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // r12
  char v8; // bl
  int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  char *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  char *v19; // rax
  char *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  char *v28; // rdx
  char *v29; // rdx
  __int64 v31; // [rsp+0h] [rbp-150h]
  char v32; // [rsp+Ch] [rbp-144h]
  __int64 v33; // [rsp+18h] [rbp-138h] BYREF
  char *v34; // [rsp+20h] [rbp-130h] BYREF
  __int64 v35; // [rsp+28h] [rbp-128h]
  char v36; // [rsp+30h] [rbp-120h] BYREF
  int v37; // [rsp+60h] [rbp-F0h]
  __int64 v38; // [rsp+68h] [rbp-E8h]
  __int64 v39; // [rsp+70h] [rbp-E0h]
  __int64 v40; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v41; // [rsp+80h] [rbp-D0h] BYREF
  char *v42; // [rsp+88h] [rbp-C8h]
  char *v43; // [rsp+90h] [rbp-C0h]
  char *v44; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v45; // [rsp+A8h] [rbp-A8h]
  char v46; // [rsp+B0h] [rbp-A0h] BYREF
  int v47; // [rsp+E0h] [rbp-70h]
  __int64 v48; // [rsp+E8h] [rbp-68h]
  __int64 v49; // [rsp+F0h] [rbp-60h]
  __int64 v50; // [rsp+F8h] [rbp-58h]
  unsigned __int64 v51; // [rsp+100h] [rbp-50h]
  char *v52; // [rsp+108h] [rbp-48h]
  char *v53; // [rsp+110h] [rbp-40h]

  v7 = a5;
  v8 = a3;
  v34 = &v36;
  v32 = a4;
  v35 = 0x600000000LL;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  sub_2332320((__int64)&v34, 1, a3, a4, a5, a6);
  v9 = *a2;
  v10 = sub_22077B0(0x10u);
  v14 = (char *)v10;
  if ( v10 )
  {
    *(_DWORD *)(v10 + 8) = v9;
    *(_QWORD *)v10 = &unk_4A13DB8;
  }
  v44 = (char *)v10;
  if ( v42 == v43 )
  {
    sub_235B010(&v41, v42, &v44);
    v14 = v44;
  }
  else
  {
    if ( v42 )
    {
      *(_QWORD *)v42 = v10;
      v42 += 8;
      goto LABEL_6;
    }
    v42 = (char *)8;
  }
  if ( v14 )
    (*(void (__fastcall **)(char *))(*(_QWORD *)v14 + 8LL))(v14);
LABEL_6:
  v44 = &v46;
  v45 = 0x600000000LL;
  if ( (_DWORD)v35 )
    sub_2303E40((__int64)&v44, &v34, v11, (unsigned int)v35, v12, v13);
  v47 = v37;
  v15 = v38;
  v38 = 0;
  v48 = v15;
  v16 = v39;
  v39 = 0;
  v49 = v16;
  v17 = v40;
  v40 = 0;
  v50 = v17;
  v18 = v41;
  v41 = 0;
  v51 = v18;
  v19 = v42;
  v42 = 0;
  v52 = v19;
  v20 = v43;
  v43 = 0;
  v53 = v20;
  v21 = sub_22077B0(0x80u);
  if ( v21 )
  {
    *(_QWORD *)(v21 + 16) = 0x600000000LL;
    *(_QWORD *)v21 = &unk_4A0B4E8;
    *(_QWORD *)(v21 + 8) = v21 + 24;
    if ( (_DWORD)v45 )
    {
      v31 = v21;
      sub_2303E40(v21 + 8, &v44, (unsigned int)v45, 0x600000000LL, v22, v23);
      v21 = v31;
    }
    *(_DWORD *)(v21 + 72) = v47;
    v24 = v48;
    v48 = 0;
    *(_QWORD *)(v21 + 80) = v24;
    v25 = v49;
    v49 = 0;
    *(_QWORD *)(v21 + 88) = v25;
    v26 = v50;
    v50 = 0;
    *(_QWORD *)(v21 + 96) = v26;
    v27 = v51;
    v51 = 0;
    *(_QWORD *)(v21 + 104) = v27;
    v28 = v52;
    v52 = 0;
    *(_QWORD *)(v21 + 112) = v28;
    v29 = v53;
    v53 = 0;
    *(_QWORD *)(v21 + 120) = v29;
  }
  v33 = v21;
  sub_2354930(a1, &v33, v8, v32, v7, 1);
  if ( v33 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v33 + 8LL))(v33);
  sub_2337B30((unsigned __int64 *)&v44);
  sub_2337B30((unsigned __int64 *)&v34);
  return a1;
}
