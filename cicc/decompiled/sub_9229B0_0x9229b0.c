// Function: sub_9229B0
// Address: 0x9229b0
//
__int64 __fastcall sub_9229B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        char a7,
        unsigned __int8 a8)
{
  int v11; // r14d
  __int64 v12; // r13
  __int64 v13; // rdi
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r11
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // r15d
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // [rsp-10h] [rbp-A0h]
  unsigned __int64 v30; // [rsp+0h] [rbp-90h]
  __int64 v31; // [rsp+8h] [rbp-88h]
  __int64 v32; // [rsp+10h] [rbp-80h]
  unsigned int v33; // [rsp+18h] [rbp-78h]
  unsigned int v35; // [rsp+28h] [rbp-68h]
  _QWORD v36[4]; // [rsp+30h] [rbp-60h] BYREF
  char v37; // [rsp+50h] [rbp-40h]
  char v38; // [rsp+51h] [rbp-3Fh]

  v11 = a8;
  if ( *(_BYTE *)(a6 + 24) != 4 )
    sub_91B8A0("unexpected field expression kind!", (_DWORD *)(a6 + 36), 1);
  v12 = *(_QWORD *)(a6 + 56);
  v13 = *(_QWORD *)(v12 + 120);
  if ( (*(_BYTE *)(v13 + 140) & 0xFB) == 8 )
  {
    v32 = a4;
    v28 = sub_8D4C10(v13, dword_4F077C4 != 2);
    a4 = v32;
    if ( (v28 & 2) != 0 )
      v11 = 1;
  }
  if ( (*(_BYTE *)(v12 + 144) & 4) != 0 )
  {
    sub_922980(a1, a2, a3, a4, v12, a5, v11);
  }
  else
  {
    v30 = a4;
    v15 = sub_917F80(*(_QWORD *)(a2 + 32) + 8LL, v12);
    v38 = 1;
    v36[0] = "tmp";
    v16 = *(_QWORD *)(a2 + 32);
    v33 = v15;
    v37 = 3;
    v17 = sub_91A390(v16 + 8, v30, 0, v30);
    v18 = sub_9213A0((unsigned int **)(a2 + 48), v17, a3, 0, v33, (__int64)v36, 7u);
    v20 = v18;
    if ( (*(_BYTE *)(v12 + 145) & 0x10) != 0 || a7 )
    {
      v31 = v18;
      v25 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL);
      v26 = sub_91A3A0(*(_QWORD *)(a2 + 32) + 8LL, *(_QWORD *)(v12 + 120), v19, v29);
      v38 = 1;
      v37 = 3;
      v36[0] = "tmp";
      v27 = sub_BCE760(v26, v25 >> 8);
      v20 = sub_920710((unsigned int **)(a2 + 48), 0x31u, v31, v27, (__int64)v36, 0, v35, 0);
    }
    v21 = a5;
    v22 = *(_QWORD *)(v12 + 128);
    if ( a5 )
    {
      while ( 1 )
      {
        v23 = v22 % v21;
        v22 = v21;
        if ( !v23 )
          break;
        v21 = v23;
      }
    }
    else
    {
      v21 = *(_QWORD *)(v12 + 128);
    }
    v24 = *(_QWORD *)(v12 + 120);
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v20;
    *(_QWORD *)(a1 + 16) = v24;
    *(_DWORD *)(a1 + 48) = v11;
    *(_DWORD *)(a1 + 24) = v21;
  }
  return a1;
}
