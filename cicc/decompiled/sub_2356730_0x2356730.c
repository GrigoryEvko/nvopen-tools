// Function: sub_2356730
// Address: 0x2356730
//
__int64 __fastcall sub_2356730(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  char v6; // r14
  char v8; // bl
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  _QWORD *v25; // rax
  _QWORD *v26; // rax
  __int64 v28; // [rsp+8h] [rbp-C8h]
  _QWORD *v29; // [rsp+18h] [rbp-B8h] BYREF
  char *v30; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v31; // [rsp+28h] [rbp-A8h]
  char v32; // [rsp+30h] [rbp-A0h] BYREF
  int v33; // [rsp+60h] [rbp-70h]
  __int64 v34; // [rsp+68h] [rbp-68h]
  __int64 v35; // [rsp+70h] [rbp-60h]
  __int64 v36; // [rsp+78h] [rbp-58h]
  __int64 v37; // [rsp+80h] [rbp-50h]
  __int64 v38; // [rsp+88h] [rbp-48h]
  __int64 v39; // [rsp+90h] [rbp-40h]

  v6 = a3;
  v8 = a5;
  v9 = *(unsigned int *)(a2 + 8);
  v30 = &v32;
  v31 = 0x600000000LL;
  if ( (_DWORD)v9 )
    sub_2303E40((__int64)&v30, (char **)a2, a3, v9, a5, a6);
  v33 = *(_DWORD *)(a2 + 64);
  v10 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a2 + 72) = 0;
  v34 = v10;
  v11 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a2 + 80) = 0;
  v35 = v11;
  v12 = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a2 + 88) = 0;
  v36 = v12;
  v13 = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a2 + 96) = 0;
  v37 = v13;
  v14 = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a2 + 104) = 0;
  v38 = v14;
  v15 = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a2 + 112) = 0;
  v39 = v15;
  v16 = sub_22077B0(0x80u);
  if ( v16 )
  {
    *(_QWORD *)(v16 + 16) = 0x600000000LL;
    *(_QWORD *)v16 = &unk_4A11F38;
    *(_QWORD *)(v16 + 8) = v16 + 24;
    if ( (_DWORD)v31 )
    {
      v28 = v16;
      sub_2303E40(v16 + 8, &v30, (unsigned int)v31, 0x600000000LL, v17, v18);
      v16 = v28;
    }
    *(_DWORD *)(v16 + 72) = v33;
    v19 = v34;
    v34 = 0;
    *(_QWORD *)(v16 + 80) = v19;
    v20 = v35;
    v35 = 0;
    *(_QWORD *)(v16 + 88) = v20;
    v21 = v36;
    v36 = 0;
    *(_QWORD *)(v16 + 96) = v21;
    v22 = v37;
    v37 = 0;
    *(_QWORD *)(v16 + 104) = v22;
    v23 = v38;
    v38 = 0;
    *(_QWORD *)(v16 + 112) = v23;
    v24 = v39;
    v39 = 0;
    *(_QWORD *)(v16 + 120) = v24;
  }
  *(_BYTE *)(a1 + 49) = a4;
  *(_QWORD *)a1 = v16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = v6;
  *(_BYTE *)(a1 + 50) = v8;
  *(_BYTE *)(a1 + 51) = 0;
  v25 = (_QWORD *)sub_22077B0(0x10u);
  if ( v25 )
    *v25 = &unk_4A0B640;
  v29 = v25;
  sub_2353900((unsigned __int64 *)(a1 + 8), (unsigned __int64 *)&v29);
  if ( v29 )
    (*(void (__fastcall **)(_QWORD *))(*v29 + 8LL))(v29);
  v26 = (_QWORD *)sub_22077B0(0x10u);
  if ( v26 )
    *v26 = &unk_4A0B680;
  v29 = v26;
  sub_2353900((unsigned __int64 *)(a1 + 8), (unsigned __int64 *)&v29);
  if ( v29 )
    (*(void (__fastcall **)(_QWORD *))(*v29 + 8LL))(v29);
  sub_2337B30((unsigned __int64 *)&v30);
  return a1;
}
