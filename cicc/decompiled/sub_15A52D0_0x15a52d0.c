// Function: sub_15A52D0
// Address: 0x15a52d0
//
_QWORD *__fastcall sub_15A52D0(_QWORD *a1, __int64 a2)
{
  char v3; // dl
  unsigned int v4; // r10d
  bool v5; // cl
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r8
  int v9; // r14d
  __int64 v10; // rbx
  __int64 v11; // r11
  __int64 v12; // rdx
  __int64 v13; // r10
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r9
  __int64 v18; // r13
  _QWORD *v19; // r15
  int v20; // ecx
  int v21; // eax
  int v22; // edx
  int v23; // eax
  __int64 v25; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+8h] [rbp-98h]
  __int64 v27; // [rsp+10h] [rbp-90h]
  __int64 v28; // [rsp+10h] [rbp-90h]
  int v29; // [rsp+10h] [rbp-90h]
  _BOOL4 v30; // [rsp+18h] [rbp-88h]
  _BOOL4 v31; // [rsp+1Ch] [rbp-84h]
  __int64 v32; // [rsp+20h] [rbp-80h]
  int v33; // [rsp+28h] [rbp-78h]
  int v34; // [rsp+2Ch] [rbp-74h]
  int v35; // [rsp+30h] [rbp-70h]
  int v36; // [rsp+34h] [rbp-6Ch]
  __int64 v37; // [rsp+38h] [rbp-68h]
  __int64 v38; // [rsp+40h] [rbp-60h]
  __int64 v39; // [rsp+48h] [rbp-58h]
  _BOOL4 v40; // [rsp+50h] [rbp-50h]
  int v41; // [rsp+54h] [rbp-4Ch]
  __int64 v42; // [rsp+58h] [rbp-48h]
  __int64 v43; // [rsp+60h] [rbp-40h]
  __int64 v44; // [rsp+68h] [rbp-38h]

  v3 = *(_BYTE *)(a2 + 40);
  v4 = *(_DWORD *)(a2 + 8);
  v36 = v3 & 3;
  v5 = (v3 & 0x10) != 0;
  v35 = *(_DWORD *)(a2 + 32);
  v34 = *(_DWORD *)(a2 + 36);
  v33 = *(_DWORD *)(a2 + 44);
  v6 = v4;
  v7 = 5LL - v4;
  v8 = 6LL - v4;
  v32 = *(_QWORD *)(a2 + 8 * (7LL - v4));
  if ( v4 > 0xA )
  {
    v44 = *(_QWORD *)(a2 + 8 * (10LL - v4));
    v38 = *(_QWORD *)(a2 + 8 * v8);
LABEL_3:
    v40 = v5;
    v37 = *(_QWORD *)(a2 + 8 * v7);
    v43 = *(_QWORD *)(a2 + 8 * (9LL - v4));
LABEL_4:
    v39 = *(_QWORD *)(a2 + 8 * (8LL - v4));
    goto LABEL_5;
  }
  v38 = *(_QWORD *)(a2 + 8 * v8);
  v37 = *(_QWORD *)(a2 + 8 * v7);
  if ( v4 == 10 )
  {
    v44 = 0;
    goto LABEL_3;
  }
  v43 = 0;
  v40 = v5;
  v44 = 0;
  if ( v4 == 9 )
    goto LABEL_4;
  v39 = 0;
LABEL_5:
  v9 = *(_DWORD *)(a2 + 24);
  v41 = *(_DWORD *)(a2 + 28);
  v30 = (v3 & 4) != 0;
  v31 = (v3 & 8) != 0;
  LODWORD(v10) = a2;
  v42 = *(_QWORD *)(a2 + 8 * (4LL - v4));
  if ( *(_BYTE *)a2 != 15 )
    v10 = *(_QWORD *)(a2 - 8LL * v4);
  v11 = *(_QWORD *)(a2 + 8 * (3LL - v4));
  if ( v11 )
  {
    v11 = sub_161E970(*(_QWORD *)(a2 + 8 * (3LL - v4)));
    v6 = *(unsigned int *)(a2 + 8);
    v13 = v12;
  }
  else
  {
    v13 = 0;
  }
  v14 = *(_QWORD *)(a2 + 8 * (2 - v6));
  if ( v14 )
  {
    v25 = v13;
    v27 = v11;
    v15 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v6)));
    v11 = v27;
    v13 = v25;
    v14 = v15;
    v6 = *(unsigned int *)(a2 + 8);
    v17 = v16;
  }
  else
  {
    v17 = 0;
  }
  v18 = *(_QWORD *)(a2 + 8 * (1 - v6));
  v19 = (_QWORD *)(*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 16) & 4) != 0 )
    v19 = (_QWORD *)*v19;
  v20 = 0;
  if ( v13 )
  {
    v26 = v14;
    v28 = v17;
    v21 = sub_161FF10(v19, v11, v13);
    v14 = v26;
    v17 = v28;
    v20 = v21;
  }
  v22 = 0;
  if ( v17 )
  {
    v29 = v20;
    v23 = sub_161FF10(v19, v14, v17);
    v20 = v29;
    v22 = v23;
  }
  *a1 = sub_15BFC70(
          (_DWORD)v19,
          v18,
          v22,
          v20,
          v10,
          v9,
          v42,
          v30,
          v31,
          v41,
          v39,
          v36,
          v35,
          v34,
          v33,
          v40,
          v37,
          v43,
          v38,
          v32,
          v44,
          2,
          1);
  return a1;
}
