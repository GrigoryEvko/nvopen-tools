// Function: sub_235CF40
// Address: 0x235cf40
//
__int64 __fastcall sub_235CF40(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // r12
  __int64 v8; // rdx
  char *v9; // rdi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  char *v16; // rax
  char *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int64 v24; // rdx
  char *v25; // rdx
  char *v26; // rdx
  __int64 v28; // [rsp+0h] [rbp-150h]
  char v29; // [rsp+Ch] [rbp-144h]
  __int64 v30; // [rsp+18h] [rbp-138h] BYREF
  char *v31; // [rsp+20h] [rbp-130h] BYREF
  __int64 v32; // [rsp+28h] [rbp-128h]
  char v33; // [rsp+30h] [rbp-120h] BYREF
  int v34; // [rsp+60h] [rbp-F0h]
  __int64 v35; // [rsp+68h] [rbp-E8h]
  __int64 v36; // [rsp+70h] [rbp-E0h]
  __int64 v37; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v38; // [rsp+80h] [rbp-D0h] BYREF
  char *v39; // [rsp+88h] [rbp-C8h]
  char *v40; // [rsp+90h] [rbp-C0h]
  char *v41; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+A8h] [rbp-A8h]
  char v43; // [rsp+B0h] [rbp-A0h] BYREF
  int v44; // [rsp+E0h] [rbp-70h]
  __int64 v45; // [rsp+E8h] [rbp-68h]
  __int64 v46; // [rsp+F0h] [rbp-60h]
  __int64 v47; // [rsp+F8h] [rbp-58h]
  unsigned __int64 v48; // [rsp+100h] [rbp-50h]
  char *v49; // [rsp+108h] [rbp-48h]
  char *v50; // [rsp+110h] [rbp-40h]

  v7 = a4;
  v31 = &v33;
  v29 = a3;
  v32 = 0x600000000LL;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  sub_2332320((__int64)&v31, 1, a3, a4, a5, a6);
  v9 = (char *)sub_22077B0(0x10u);
  if ( v9 )
    *(_QWORD *)v9 = &unk_4A13D38;
  v41 = v9;
  if ( v39 == v40 )
  {
    sub_235B010(&v38, v39, &v41);
    v9 = v41;
  }
  else
  {
    if ( v39 )
    {
      *(_QWORD *)v39 = v9;
      v39 += 8;
      goto LABEL_6;
    }
    v39 = (char *)8;
  }
  if ( v9 )
    (*(void (__fastcall **)(char *))(*(_QWORD *)v9 + 8LL))(v9);
LABEL_6:
  v41 = &v43;
  v42 = 0x600000000LL;
  if ( (_DWORD)v32 )
    sub_2303E40((__int64)&v41, &v31, v8, (unsigned int)v32, v10, v11);
  v44 = v34;
  v12 = v35;
  v35 = 0;
  v45 = v12;
  v13 = v36;
  v36 = 0;
  v46 = v13;
  v14 = v37;
  v37 = 0;
  v47 = v14;
  v15 = v38;
  v38 = 0;
  v48 = v15;
  v16 = v39;
  v39 = 0;
  v49 = v16;
  v17 = v40;
  v40 = 0;
  v50 = v17;
  v18 = sub_22077B0(0x80u);
  if ( v18 )
  {
    *(_QWORD *)(v18 + 16) = 0x600000000LL;
    *(_QWORD *)v18 = &unk_4A0B4E8;
    *(_QWORD *)(v18 + 8) = v18 + 24;
    if ( (_DWORD)v42 )
    {
      v28 = v18;
      sub_2303E40(v18 + 8, &v41, (unsigned int)v42, 0x600000000LL, v19, v20);
      v18 = v28;
    }
    *(_DWORD *)(v18 + 72) = v44;
    v21 = v45;
    v45 = 0;
    *(_QWORD *)(v18 + 80) = v21;
    v22 = v46;
    v46 = 0;
    *(_QWORD *)(v18 + 88) = v22;
    v23 = v47;
    v47 = 0;
    *(_QWORD *)(v18 + 96) = v23;
    v24 = v48;
    v48 = 0;
    *(_QWORD *)(v18 + 104) = v24;
    v25 = v49;
    v49 = 0;
    *(_QWORD *)(v18 + 112) = v25;
    v26 = v50;
    v50 = 0;
    *(_QWORD *)(v18 + 120) = v26;
  }
  v30 = v18;
  sub_2354930(a1, &v30, a2, v29, v7, 1);
  if ( v30 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
  sub_2337B30((unsigned __int64 *)&v41);
  sub_2337B30((unsigned __int64 *)&v31);
  return a1;
}
