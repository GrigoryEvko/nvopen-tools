// Function: sub_235AE30
// Address: 0x235ae30
//
void __fastcall sub_235AE30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // [rsp+8h] [rbp-A8h] BYREF
  char *v29; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v30; // [rsp+18h] [rbp-98h]
  char v31; // [rsp+20h] [rbp-90h] BYREF
  int v32; // [rsp+50h] [rbp-60h]
  __int64 v33; // [rsp+58h] [rbp-58h]
  __int64 v34; // [rsp+60h] [rbp-50h]
  __int64 v35; // [rsp+68h] [rbp-48h]
  __int64 v36; // [rsp+70h] [rbp-40h]
  __int64 v37; // [rsp+78h] [rbp-38h]
  __int64 v38; // [rsp+80h] [rbp-30h]

  sub_2332320(a1, 0, a3, a4, a5, a6);
  v9 = *(unsigned int *)(a2 + 8);
  v29 = &v31;
  v30 = 0x600000000LL;
  if ( (_DWORD)v9 )
    sub_2303E40((__int64)&v29, (char **)a2, v9, v6, v7, v8);
  v32 = *(_DWORD *)(a2 + 64);
  v10 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a2 + 72) = 0;
  v33 = v10;
  v11 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a2 + 80) = 0;
  v34 = v11;
  v12 = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a2 + 88) = 0;
  v35 = v12;
  v13 = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a2 + 96) = 0;
  v36 = v13;
  v14 = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a2 + 104) = 0;
  v37 = v14;
  v15 = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a2 + 112) = 0;
  v38 = v15;
  v16 = (_QWORD *)sub_22077B0(0x80u);
  v21 = (unsigned __int64)v16;
  if ( v16 )
  {
    *v16 = &unk_4A11F38;
    v16[1] = v16 + 3;
    v16[2] = 0x600000000LL;
    if ( (_DWORD)v30 )
      sub_2303E40((__int64)(v16 + 1), &v29, v17, v18, v19, v20);
    *(_DWORD *)(v21 + 72) = v32;
    v22 = v33;
    v33 = 0;
    *(_QWORD *)(v21 + 80) = v22;
    v23 = v34;
    v34 = 0;
    *(_QWORD *)(v21 + 88) = v23;
    v24 = v35;
    v35 = 0;
    *(_QWORD *)(v21 + 96) = v24;
    v25 = v36;
    v36 = 0;
    *(_QWORD *)(v21 + 104) = v25;
    v26 = v37;
    v37 = 0;
    *(_QWORD *)(v21 + 112) = v26;
    v27 = v38;
    v38 = 0;
    *(_QWORD *)(v21 + 120) = v27;
  }
  v28 = v21;
  sub_235ACD0((unsigned __int64 *)(a1 + 72), &v28);
  if ( v28 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v28 + 8LL))(v28);
  sub_2337B30((unsigned __int64 *)&v29);
}
