// Function: sub_12ADE80
// Address: 0x12ade80
//
__int64 __fastcall sub_12ADE80(__int64 a1, _QWORD *a2, unsigned int a3, int a4, char a5, __int64 a6)
{
  unsigned int v9; // ebx
  unsigned __int64 *v10; // r8
  __int64 v11; // r15
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v19; // r9
  __int64 v20; // rsi
  char *v21; // rax
  __int64 v23; // [rsp+8h] [rbp-88h]
  __int64 v24; // [rsp+8h] [rbp-88h]
  __int64 v25; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v26[2]; // [rsp+20h] [rbp-70h] BYREF
  char *v27; // [rsp+30h] [rbp-60h]
  _QWORD v28[2]; // [rsp+40h] [rbp-50h] BYREF
  char *v29; // [rsp+50h] [rbp-40h]
  char *v30; // [rsp+58h] [rbp-38h]

  v9 = (unsigned __int8)a4 << 16;
  LOBYTE(v9) = (16 * a5) | 5;
  v10 = *(unsigned __int64 **)(a6 + 16);
  v23 = (__int64)v10;
  v11 = v10[2];
  v12 = sub_127A030(a2[4] + 8LL, *v10, 0);
  v13 = (_QWORD *)a2[4];
  v25 = v12;
  v14 = sub_126A190(v13, a3, (__int64)&v25, 1u);
  v15 = sub_1643350(a2[5]);
  v16 = sub_159C470(v15, v9, 0);
  if ( a4 == 14 )
  {
    v19 = *(_QWORD *)(v11 + 16);
    v20 = v23;
    v28[0] = v16;
    v24 = v19;
    v28[1] = sub_128F980((__int64)a2, v20);
    v29 = sub_128F980((__int64)a2, v11);
    v21 = sub_128F980((__int64)a2, v24);
    LOWORD(v27) = 257;
    v30 = v21;
    v17 = sub_1285290(a2 + 6, *(_QWORD *)(v14 + 24), v14, (int)v28, 4, (__int64)v26, 0);
  }
  else
  {
    v26[0] = v16;
    v26[1] = sub_128F980((__int64)a2, v23);
    v27 = sub_128F980((__int64)a2, v11);
    LOWORD(v29) = 257;
    v17 = sub_1285290(a2 + 6, *(_QWORD *)(v14 + 24), v14, (int)v26, 3, (__int64)v28, 0);
  }
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v17;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
