// Function: sub_3746A00
// Address: 0x3746a00
//
__int64 __fastcall sub_3746A00(__int64 *a1, unsigned int a2, __int64 a3)
{
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int v11; // ebx
  unsigned int v12; // r15d
  unsigned int v13; // r13d
  __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  char v17; // si
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  char v22; // al
  unsigned __int64 v23; // rcx
  __int64 v24; // rdx
  char v25; // si
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdi
  unsigned __int64 v29; // rdx
  char v30; // al
  __int64 (*v31)(); // rax
  __int64 v32; // r8
  __int64 v33; // rcx
  unsigned __int64 v34; // [rsp+8h] [rbp-98h]
  unsigned __int64 v35; // [rsp+8h] [rbp-98h]
  unsigned int v36; // [rsp+10h] [rbp-90h] BYREF
  __int64 v37; // [rsp+18h] [rbp-88h]
  __int64 v38; // [rsp+20h] [rbp-80h]
  __int64 v39; // [rsp+28h] [rbp-78h]
  __int64 v40; // [rsp+30h] [rbp-70h]
  __int64 v41; // [rsp+38h] [rbp-68h]
  __int16 v42; // [rsp+40h] [rbp-60h] BYREF
  __int64 v43; // [rsp+48h] [rbp-58h]
  __int64 v44; // [rsp+50h] [rbp-50h] BYREF
  __int64 v45; // [rsp+58h] [rbp-48h]
  __int64 v46; // [rsp+60h] [rbp-40h]
  __int64 v47; // [rsp+68h] [rbp-38h]

  v6 = sub_3746830(a1, a3);
  if ( !v6 )
    return 0;
  v11 = v6;
  v12 = v6;
  v36 = sub_30097B0(*(_QWORD *)(a3 + 8), 0, v7, v8, v9);
  v13 = v36;
  v37 = v14;
  if ( (_WORD)a2 == (_WORD)v36 )
  {
    if ( (_WORD)a2 )
      return v12;
    if ( !v37 )
      goto LABEL_38;
    v45 = 0;
    LOWORD(v44) = 0;
    goto LABEL_5;
  }
  LOWORD(v44) = a2;
  v45 = 0;
  if ( !(_WORD)a2 )
  {
LABEL_5:
    v40 = sub_3007260((__int64)&v44);
    v15 = v40;
    v41 = v16;
    v17 = v16;
    goto LABEL_6;
  }
  if ( (_WORD)a2 == 1 || (unsigned __int16)(a2 - 504) <= 7u )
    goto LABEL_44;
  v15 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a2 - 16];
  v17 = byte_444C4A0[16 * (unsigned __int16)a2 - 8];
LABEL_6:
  if ( (_WORD)v36 )
  {
    if ( (_WORD)v36 == 1 || (unsigned __int16)(v36 - 504) <= 7u )
      goto LABEL_44;
    v21 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v36 - 16];
    v22 = byte_444C4A0[16 * (unsigned __int16)v36 - 8];
  }
  else
  {
    v34 = v15;
    v18 = sub_3007260((__int64)&v36);
    v15 = v34;
    v20 = v19;
    v38 = v18;
    v21 = v18;
    v39 = v20;
    v22 = v20;
  }
  if ( (!v22 || v17) && v15 > v21 )
  {
    v31 = *(__int64 (**)())(*a1 + 64);
    if ( v31 == sub_3740EE0 )
      return 0;
    v32 = v11;
    v33 = 213;
    return ((unsigned int (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, __int64))v31)(a1, v13, a2, v33, v32);
  }
  if ( (_WORD)a2 == (_WORD)v13 )
  {
    if ( (_WORD)a2 )
      return v12;
LABEL_38:
    if ( !v37 )
      return v12;
  }
  v42 = a2;
  v43 = 0;
  if ( (_WORD)a2 )
  {
    if ( (_WORD)a2 == 1 || (unsigned __int16)(a2 - 504) <= 7u )
      goto LABEL_44;
    v23 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a2 - 16];
    v25 = byte_444C4A0[16 * (unsigned __int16)a2 - 8];
  }
  else
  {
    v46 = sub_3007260((__int64)&v42);
    v23 = v46;
    v47 = v24;
    v25 = v24;
  }
  if ( (_WORD)v13 )
  {
    if ( (_WORD)v13 != 1 && (unsigned __int16)(v13 - 504) > 7u )
    {
      v29 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v13 - 16];
      v30 = byte_444C4A0[16 * (unsigned __int16)v13 - 8];
      goto LABEL_15;
    }
LABEL_44:
    BUG();
  }
  v35 = v23;
  v26 = sub_3007260((__int64)&v36);
  v23 = v35;
  v28 = v27;
  v44 = v26;
  v29 = v26;
  v45 = v28;
  v30 = v28;
LABEL_15:
  if ( !v30 && v25 || v23 >= v29 )
    return v12;
  v31 = *(__int64 (**)())(*a1 + 64);
  if ( v31 == sub_3740EE0 )
    return 0;
  v32 = v11;
  v33 = 216;
  return ((unsigned int (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, __int64))v31)(a1, v13, a2, v33, v32);
}
