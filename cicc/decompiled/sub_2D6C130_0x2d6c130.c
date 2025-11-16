// Function: sub_2D6C130
// Address: 0x2d6c130
//
__int64 __fastcall sub_2D6C130(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  unsigned int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  __int64 *v11; // rdx
  unsigned __int16 v12; // r15
  __int64 v13; // rdx
  char v14; // r14
  unsigned __int16 v15; // bx
  char v16; // al
  unsigned __int64 v17; // r14
  __int64 v18; // rdx
  char v19; // cl
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  unsigned __int64 v23; // rdx
  char v24; // al
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 (__fastcall *v28)(__int64, __int64, unsigned int, __int64); // rbx
  __int64 v29; // rax
  __int64 (__fastcall *v30)(__int64, __int64, unsigned int, __int64); // r15
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  char v34; // [rsp+7h] [rbp-99h]
  __int64 v35; // [rsp+8h] [rbp-98h]
  __int64 v36; // [rsp+10h] [rbp-90h] BYREF
  __int64 v37; // [rsp+18h] [rbp-88h]
  __int64 v38; // [rsp+20h] [rbp-80h] BYREF
  __int64 v39; // [rsp+28h] [rbp-78h]
  __int64 v40; // [rsp+30h] [rbp-70h]
  __int64 v41; // [rsp+38h] [rbp-68h]
  __int64 v42; // [rsp+40h] [rbp-60h]
  __int64 v43; // [rsp+48h] [rbp-58h]
  unsigned __int16 v44; // [rsp+50h] [rbp-50h] BYREF
  __int64 v45; // [rsp+58h] [rbp-48h]
  __int64 v46; // [rsp+60h] [rbp-40h]

  if ( *(_BYTE *)a1 == 79 )
  {
    v5 = *(_QWORD *)(a1 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
      v5 = **(_QWORD **)(v5 + 16);
    v6 = *(_DWORD *)(v5 + 8);
    v7 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL);
    v8 = v6 >> 8;
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
      v7 = **(_QWORD **)(v7 + 16);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 992LL))(
            a2,
            *(_DWORD *)(v7 + 8) >> 8,
            v8) )
      return 0;
  }
  v9 = sub_2D5BAE0(a2, a3, *(__int64 **)(*(_QWORD *)(a1 - 32) + 8LL), 0);
  v37 = v10;
  v11 = *(__int64 **)(a1 + 8);
  LODWORD(v36) = v9;
  v12 = v9;
  LODWORD(v38) = sub_2D5BAE0(a2, a3, v11, 0);
  v39 = v13;
  if ( !(_WORD)v36 )
  {
    v14 = sub_3007070(&v36);
    goto LABEL_9;
  }
  v14 = (unsigned __int16)(v12 - 17) <= 0x6Cu || (unsigned __int16)(v12 - 2) <= 7u;
  if ( v14 )
  {
LABEL_9:
    v15 = v38;
    if ( (_WORD)v38 )
      goto LABEL_10;
LABEL_24:
    v16 = sub_3007070(&v38);
    goto LABEL_12;
  }
  v15 = v38;
  v14 = (unsigned __int16)(v12 - 176) <= 0x1Fu;
  if ( !(_WORD)v38 )
    goto LABEL_24;
LABEL_10:
  v16 = (unsigned __int16)(v15 - 17) <= 0x6Cu || (unsigned __int16)(v15 - 2) <= 7u;
  if ( !v16 )
    v16 = (unsigned __int16)(v15 - 176) <= 0x1Fu;
LABEL_12:
  if ( v16 != v14 )
    return 0;
  v35 = v39;
  if ( v15 == v12 )
  {
    if ( v15 || v39 == v37 )
      goto LABEL_26;
    v44 = 0;
    v45 = v39;
    goto LABEL_15;
  }
  v44 = v15;
  v45 = v39;
  if ( !v15 )
  {
LABEL_15:
    v42 = sub_3007260(&v44);
    v17 = v42;
    v43 = v18;
    v19 = v18;
    goto LABEL_16;
  }
  if ( v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
    goto LABEL_47;
  v17 = *(_QWORD *)&byte_444C4A0[16 * v15 - 16];
  v19 = byte_444C4A0[16 * v15 - 8];
LABEL_16:
  if ( !v12 )
  {
    v34 = v19;
    v20 = sub_3007260(&v36);
    v19 = v34;
    v22 = v21;
    v40 = v20;
    v23 = v20;
    v41 = v22;
    v24 = v22;
    goto LABEL_18;
  }
  if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
LABEL_47:
    BUG();
  v23 = *(_QWORD *)&byte_444C4A0[16 * v12 - 16];
  v24 = byte_444C4A0[16 * v12 - 8];
LABEL_18:
  if ( (!v24 || v19) && v23 < v17 )
    return 0;
LABEL_26:
  v26 = sub_BD5C60(a1);
  sub_2FE6CC0(&v44, a2, v26, v36, v37);
  if ( (_BYTE)v44 == 1 )
  {
    v30 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a2 + 592LL);
    v31 = sub_BD5C60(a1);
    if ( v30 == sub_2D56A50 )
    {
      sub_2FE6CC0(&v44, a2, v31, v36, v37);
      v12 = v45;
      LOWORD(v36) = v45;
      v37 = v46;
    }
    else
    {
      LODWORD(v36) = v30(a2, v31, v36, v37);
      v12 = v36;
      v37 = v33;
    }
  }
  v27 = sub_BD5C60(a1);
  sub_2FE6CC0(&v44, a2, v27, v38, v39);
  if ( (_BYTE)v44 == 1 )
  {
    v28 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a2 + 592LL);
    v29 = sub_BD5C60(a1);
    if ( v28 == sub_2D56A50 )
    {
      sub_2FE6CC0(&v44, a2, v29, v38, v39);
      v15 = v45;
      v35 = v46;
      LOWORD(v38) = v45;
      v39 = v46;
    }
    else
    {
      LODWORD(v38) = v28(a2, v29, v38, v39);
      v15 = v38;
      v39 = v32;
      v35 = v32;
    }
  }
  if ( v12 != v15 || !v12 && v37 != v35 )
    return 0;
  return sub_2D6BC60(a1);
}
