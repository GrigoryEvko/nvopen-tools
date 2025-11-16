// Function: sub_1290BA0
// Address: 0x1290ba0
//
__int64 __fastcall sub_1290BA0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  char v5; // al
  __int64 v6; // r13
  __int64 *v7; // r14
  __int64 *v8; // rax
  unsigned __int64 v9; // rsi
  __int64 *v10; // r15
  __int64 v11; // rdx
  unsigned int v12; // eax
  unsigned int v13; // ebx
  int v14; // eax
  __int64 v15; // r8
  _BOOL4 v16; // edx
  int v17; // r14d
  _BOOL4 v18; // ecx
  int v20; // r15d
  __int64 v21; // rdi
  int v22; // ebx
  __int64 v23; // rax
  int v24; // r14d
  int v25; // r13d
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rdi
  _BOOL4 v29; // r15d
  unsigned int v30; // r14d
  __int64 v31; // rax
  __int64 v32; // rdi
  _BOOL4 v33; // ebx
  unsigned int v34; // eax
  char v35; // al
  __int64 v36; // rdi
  char v37; // al
  int v38; // eax
  __int64 v39; // [rsp-10h] [rbp-80h]
  _QWORD *v40; // [rsp+8h] [rbp-68h]
  const __m128i *v41; // [rsp+10h] [rbp-60h]
  _BOOL4 v42; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  __int64 v44; // [rsp+18h] [rbp-58h]
  __int64 v45; // [rsp+18h] [rbp-58h]
  const char *v46; // [rsp+20h] [rbp-50h] BYREF
  char v47; // [rsp+30h] [rbp-40h]
  char v48; // [rsp+31h] [rbp-3Fh]

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 8);
  switch ( v5 )
  {
    case 2:
      v41 = *(const __m128i **)(a2 + 56);
      v7 = (__int64 *)sub_127F650((__int64)a1, v41, *(_QWORD *)(v6 + 120));
      v8 = (__int64 *)sub_12A2A10(a1, v6);
      v9 = *(_QWORD *)(v6 + 120);
      v10 = v8;
      v43 = *v8;
      if ( *v7 == sub_127A030(a1[4] + 8LL, v9, 0) )
      {
        v32 = *(_QWORD *)(v6 + 120);
        v33 = 0;
        if ( (*(_BYTE *)(v32 + 140) & 0xFB) == 8 )
        {
          v9 = dword_4F077C4 != 2;
          v33 = (sub_8D4C10(v32, v9) & 2) != 0;
        }
        v34 = sub_127C800(v6, v9, v11);
        return sub_12A61B0(a1, v7, v10, v34, v33);
      }
      else
      {
        v12 = sub_127C800(v6, v9, v11);
        v48 = 1;
        v47 = 3;
        v13 = v12;
        v46 = "consttmp";
        v40 = sub_127FC40(a1, *v7, (__int64)&v46, v12, 0);
        sub_12A61B0(a1, v7, v40, v13, 0);
        v14 = sub_128B420((__int64)a1, v40, 0, v43, 0, 0, (const __m128i *)v41[4].m128i_i32);
        v15 = *(_QWORD *)(v6 + 120);
        v16 = 0;
        v17 = v14;
        v18 = 0;
        if ( (*(_BYTE *)(v15 + 140) & 0xFB) == 8 )
        {
          v45 = *(_QWORD *)(v6 + 120);
          v35 = sub_8D4C10(v45, dword_4F077C4 != 2);
          v36 = *(_QWORD *)(v6 + 120);
          v15 = v45;
          v18 = 0;
          v16 = (v35 & 2) != 0;
          if ( (*(_BYTE *)(v36 + 140) & 0xFB) == 8 )
          {
            v42 = (v35 & 2) != 0;
            v37 = sub_8D4C10(v36, dword_4F077C4 != 2);
            v16 = v42;
            v15 = v45;
            v18 = (v37 & 2) != 0;
          }
        }
        sub_12A6300((_DWORD)a1, (_DWORD)v10, v13, v18, v17, v13, v16, v15);
        return v39;
      }
    case 3:
      v28 = *(_QWORD *)(v6 + 120);
      v29 = 0;
      if ( (*(_BYTE *)(v28 + 140) & 0xFB) == 8 )
      {
        a2 = dword_4F077C4 != 2;
        v29 = (sub_8D4C10(v28, a2) & 2) != 0;
      }
      v30 = sub_127C800(v6, a2, a3);
      v31 = sub_12A2A10(a1, v6);
      return sub_12A6C40(a1, *(_QWORD *)(v4 + 56), v31, v30, v29);
    case 1:
      v20 = (_DWORD)a1 + 48;
      v21 = *(_QWORD *)(v6 + 120);
      if ( *(char *)(v21 + 142) < 0 )
      {
        v22 = *(_DWORD *)(v21 + 136);
      }
      else
      {
        if ( *(_BYTE *)(v21 + 140) != 12 )
        {
          v22 = *(_DWORD *)(v21 + 136);
LABEL_11:
          v44 = *(_QWORD *)(v21 + 128);
          v23 = sub_1643330(a1[5]);
          v24 = sub_15A06D0(v23);
          v25 = sub_12A2A10(a1, v6);
          v26 = sub_1643360(a1[9]);
          v27 = sub_159C470(v26, v44, 0);
          return sub_15E7280(v20, v25, v24, v27, v22, 0, 0, 0, 0);
        }
        v38 = sub_8D4AB0(v21);
        v21 = *(_QWORD *)(v6 + 120);
        v22 = v38;
      }
      while ( *(_BYTE *)(v21 + 140) == 12 )
        v21 = *(_QWORD *)(v21 + 160);
      goto LABEL_11;
    default:
      sub_127B550("unsupported dynamic initialization variant!", (_DWORD *)(v6 + 64), 1);
  }
}
