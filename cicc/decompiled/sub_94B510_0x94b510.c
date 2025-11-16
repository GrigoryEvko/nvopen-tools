// Function: sub_94B510
// Address: 0x94b510
//
_BYTE *__fastcall sub_94B510(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  int v5; // eax
  int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  char v10; // al
  int v11; // ebx
  __int64 v12; // rax
  _BYTE *v13; // r15
  unsigned int *v14; // r14
  unsigned int *v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 (__fastcall *v19)(__int64, _BYTE *, _BYTE *, _BYTE *); // rax
  __int64 v20; // rax
  __int64 v22; // rax
  int v23; // r9d
  _BYTE *v24; // r14
  unsigned int *v25; // rax
  unsigned int *v26; // r15
  unsigned int *i; // rbx
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rbx
  __int64 v31; // rdi
  __int64 v32; // rax
  int v33; // r15d
  __int64 v34; // rax
  unsigned int *v35; // rbx
  unsigned int *v36; // r12
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // [rsp+28h] [rbp-108h]
  __int64 v40; // [rsp+30h] [rbp-100h]
  _BYTE *v42; // [rsp+40h] [rbp-F0h]
  unsigned __int16 v43; // [rsp+4Eh] [rbp-E2h]
  _BYTE *v44; // [rsp+58h] [rbp-D8h]
  __int64 v45; // [rsp+60h] [rbp-D0h]
  __int64 v46; // [rsp+68h] [rbp-C8h]
  char v47[32]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v48; // [rsp+90h] [rbp-A0h]
  _BYTE v49[32]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v50; // [rsp+C0h] [rbp-70h]
  _BYTE v51[32]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v52; // [rsp+F0h] [rbp-40h]

  if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 > 1 )
  {
    v30 = a1 + 48;
    v31 = *(_QWORD *)(a1 + 96);
    v50 = 257;
    v32 = sub_AA4E30(v31);
    v33 = (unsigned __int8)sub_AE5020(v32, a2);
    v52 = 257;
    v34 = sub_BD2C40(80, unk_3F10A14);
    v4 = v34;
    if ( v34 )
      sub_B4D190(v34, a2, a3, (unsigned int)v51, 0, v33, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
      *(_QWORD *)(a1 + 136),
      v4,
      v49,
      *(_QWORD *)(v30 + 56),
      *(_QWORD *)(v30 + 64));
    v35 = *(unsigned int **)(a1 + 48);
    v36 = &v35[4 * *(unsigned int *)(a1 + 56)];
    while ( v36 != v35 )
    {
      v37 = *((_QWORD *)v35 + 1);
      v38 = *v35;
      v35 += 4;
      sub_B99FD0(v4, v38, v37);
    }
  }
  else
  {
    v4 = sub_ACA8A0(a2);
    v40 = sub_BCB2D0(*(_QWORD *)(a1 + 40));
    v45 = *(_QWORD *)(a2 + 24);
    if ( *(_DWORD *)(a2 + 32) )
    {
      v39 = *(unsigned int *)(a2 + 32);
      v46 = 0;
      v44 = (_BYTE *)v4;
      while ( 1 )
      {
        v52 = 257;
        v5 = sub_94B2B0((unsigned int **)(a1 + 48), v45, a3, v46, (__int64)v51);
        v48 = 257;
        v6 = v5;
        v7 = sub_AD64C0(v40, v46, 0);
        v8 = *(_QWORD *)(a1 + 96);
        v50 = 257;
        v42 = (_BYTE *)v7;
        v9 = sub_AA4E30(v8);
        v10 = sub_AE5020(v9, v45);
        v11 = v43;
        v52 = 257;
        LOBYTE(v11) = v10;
        v43 = v11;
        v12 = sub_BD2C40(80, unk_3F10A14);
        v13 = (_BYTE *)v12;
        if ( v12 )
          sub_B4D190(v12, v45, v6, (unsigned int)v51, 0, v11, 0, 0);
        (*(void (__fastcall **)(_QWORD, _BYTE *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
          *(_QWORD *)(a1 + 136),
          v13,
          v49,
          *(_QWORD *)(a1 + 104),
          *(_QWORD *)(a1 + 112));
        v14 = *(unsigned int **)(a1 + 48);
        v15 = &v14[4 * *(unsigned int *)(a1 + 56)];
        while ( v15 != v14 )
        {
          v16 = *((_QWORD *)v14 + 1);
          v17 = *v14;
          v14 += 4;
          sub_B99FD0(v13, v17, v16);
        }
        v18 = *(_QWORD *)(a1 + 128);
        v19 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, _BYTE *))(*(_QWORD *)v18 + 104LL);
        if ( v19 == sub_948040 )
        {
          if ( *v44 > 0x15u || *v13 > 0x15u || *v42 > 0x15u )
          {
LABEL_18:
            v52 = 257;
            v22 = sub_BD2C40(72, 3);
            v24 = (_BYTE *)v22;
            if ( v22 )
              sub_B4DFA0(v22, (_DWORD)v44, (_DWORD)v13, (_DWORD)v42, (unsigned int)v51, v23, 0, 0);
            (*(void (__fastcall **)(_QWORD, _BYTE *, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
              *(_QWORD *)(a1 + 136),
              v24,
              v47,
              *(_QWORD *)(a1 + 104),
              *(_QWORD *)(a1 + 112));
            v25 = *(unsigned int **)(a1 + 48);
            v26 = &v25[4 * *(unsigned int *)(a1 + 56)];
            for ( i = v25; v26 != i; i += 4 )
            {
              v28 = *((_QWORD *)i + 1);
              v29 = *i;
              sub_B99FD0(v24, v29, v28);
            }
            v44 = v24;
            goto LABEL_15;
          }
          v20 = sub_AD5A90(v44, v13, v42, 0);
        }
        else
        {
          v20 = v19(v18, v44, v13, v42);
        }
        if ( !v20 )
          goto LABEL_18;
        v44 = (_BYTE *)v20;
LABEL_15:
        if ( ++v46 == v39 )
          return v44;
      }
    }
  }
  return (_BYTE *)v4;
}
