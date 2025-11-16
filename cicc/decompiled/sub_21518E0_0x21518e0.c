// Function: sub_21518E0
// Address: 0x21518e0
//
__int64 __fastcall sub_21518E0(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 i; // rbx
  __int64 v8; // rax
  _DWORD *v9; // rdx
  __int64 v10; // r10
  const char **v11; // r9
  const char *v12; // rax
  size_t v13; // rdx
  __int64 v14; // r9
  char *v15; // rsi
  _WORD *v16; // rdi
  unsigned __int64 v17; // rax
  const char *v18; // rax
  size_t v19; // rdx
  __int64 v20; // r9
  char *v21; // rsi
  _WORD *v22; // rdi
  unsigned __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int64 v25; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // [rsp+0h] [rbp-140h]
  __int64 v33; // [rsp+0h] [rbp-140h]
  __int64 v34; // [rsp+10h] [rbp-130h]
  __int64 v35; // [rsp+10h] [rbp-130h]
  size_t v36; // [rsp+10h] [rbp-130h]
  size_t v37; // [rsp+10h] [rbp-130h]
  _QWORD v38[2]; // [rsp+20h] [rbp-120h] BYREF
  _QWORD *v39; // [rsp+30h] [rbp-110h] BYREF
  __int16 v40; // [rsp+40h] [rbp-100h]
  const char *v41; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v42; // [rsp+58h] [rbp-E8h]
  __int64 v43; // [rsp+60h] [rbp-E0h]
  _DWORD *v44; // [rsp+68h] [rbp-D8h]
  int v45; // [rsp+70h] [rbp-D0h]
  unsigned __int64 *v46; // [rsp+78h] [rbp-C8h]
  unsigned __int64 v47[2]; // [rsp+80h] [rbp-C0h] BYREF
  _WORD v48[88]; // [rsp+90h] [rbp-B0h] BYREF

  a1[105] = a2[2];
  sub_396F5A0();
  sub_3979400(a1);
  v3 = a1[32];
  v41 = "}\n";
  v48[0] = 261;
  v42 = 2;
  v47[0] = (unsigned __int64)&v41;
  sub_38DD5A0(v3, v47);
  v4 = *a2;
  v5 = *(_QWORD *)(*a2 + 40);
  v6 = *(_QWORD *)(v5 + 48);
  for ( i = v5 + 40; i != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    while ( 1 )
    {
      if ( !v6 )
        BUG();
      v8 = *(_QWORD *)(v6 - 72);
      if ( !v8 )
        BUG();
      if ( (*(_BYTE *)(v8 + 16) != 5
         || *(_WORD *)(v8 + 18) != 47
         || (v8 = *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF))) != 0)
        && v8 == v4 )
      {
        v47[0] = (unsigned __int64)v48;
        v47[1] = 0x8000000000LL;
        v45 = 1;
        v44 = 0;
        v41 = (const char *)&unk_49EFC48;
        v43 = 0;
        v46 = v47;
        v42 = 0;
        sub_16E7A40((__int64)&v41, 0, 0, 0);
        sub_2151550(a1, v6 - 48, v4, (__int64)&v41);
        v9 = v44;
        v10 = v6 - 48;
        if ( (unsigned __int64)(v43 - (_QWORD)v44) <= 6 )
        {
          v27 = sub_16E7EE0((__int64)&v41, ".alias ", 7u);
          v10 = v6 - 48;
          v11 = (const char **)v27;
        }
        else
        {
          *v44 = 1768710446;
          v11 = &v41;
          *((_WORD *)v9 + 2) = 29537;
          *((_BYTE *)v9 + 6) = 32;
          v44 = (_DWORD *)((char *)v44 + 7);
        }
        v34 = (__int64)v11;
        v12 = sub_1649960(v10);
        v14 = v34;
        v15 = (char *)v12;
        v16 = *(_WORD **)(v34 + 24);
        v17 = *(_QWORD *)(v34 + 16) - (_QWORD)v16;
        if ( v13 > v17 )
        {
          v29 = sub_16E7EE0(v34, v15, v13);
          v16 = *(_WORD **)(v29 + 24);
          v14 = v29;
          v17 = *(_QWORD *)(v29 + 16) - (_QWORD)v16;
        }
        else if ( v13 )
        {
          v32 = v34;
          v36 = v13;
          memcpy(v16, v15, v13);
          v14 = v32;
          v30 = *(_QWORD *)(v32 + 16);
          v16 = (_WORD *)(*(_QWORD *)(v32 + 24) + v36);
          *(_QWORD *)(v32 + 24) = v16;
          v17 = v30 - (_QWORD)v16;
        }
        if ( v17 <= 1 )
        {
          v14 = sub_16E7EE0(v14, ", ", 2u);
        }
        else
        {
          *v16 = 8236;
          *(_QWORD *)(v14 + 24) += 2LL;
        }
        v35 = v14;
        v18 = sub_1649960(v4);
        v20 = v35;
        v21 = (char *)v18;
        v22 = *(_WORD **)(v35 + 24);
        v23 = *(_QWORD *)(v35 + 16) - (_QWORD)v22;
        if ( v23 < v19 )
        {
          v28 = sub_16E7EE0(v35, v21, v19);
          v22 = *(_WORD **)(v28 + 24);
          v20 = v28;
          v23 = *(_QWORD *)(v28 + 16) - (_QWORD)v22;
        }
        else if ( v19 )
        {
          v33 = v35;
          v37 = v19;
          memcpy(v22, v21, v19);
          v20 = v33;
          v31 = *(_QWORD *)(v33 + 16);
          v22 = (_WORD *)(*(_QWORD *)(v33 + 24) + v37);
          *(_QWORD *)(v33 + 24) = v22;
          v23 = v31 - (_QWORD)v22;
        }
        if ( v23 <= 1 )
        {
          sub_16E7EE0(v20, ";\n", 2u);
        }
        else
        {
          *v22 = 2619;
          *(_QWORD *)(v20 + 24) += 2LL;
        }
        v24 = a1[32];
        v25 = *v46;
        v38[1] = *((unsigned int *)v46 + 2);
        v40 = 261;
        v38[0] = v25;
        v39 = v38;
        sub_38DD5A0(v24, &v39);
        v41 = (const char *)&unk_49EFD28;
        sub_16E7960((__int64)&v41);
        if ( (_WORD *)v47[0] != v48 )
          break;
      }
      v6 = *(_QWORD *)(v6 + 8);
      if ( i == v6 )
        return 0;
    }
    _libc_free(v47[0]);
  }
  return 0;
}
