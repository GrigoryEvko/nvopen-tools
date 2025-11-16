// Function: sub_3380740
// Address: 0x3380740
//
__int64 __fastcall sub_3380740(__int64 a1, int *a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned int v8; // ecx
  __int64 v9; // rdx
  int *v10; // r8
  __int64 v11; // rax
  int v12; // r15d
  int v13; // eax
  __int64 v14; // rsi
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r14
  unsigned int v21; // edx
  __int64 v22; // r15
  int v24; // edx
  int v25; // r9d
  __int64 v26; // [rsp+0h] [rbp-120h]
  __int64 v27; // [rsp+10h] [rbp-110h] BYREF
  int v28; // [rsp+18h] [rbp-108h]
  __int64 v29; // [rsp+20h] [rbp-100h] BYREF
  int v30; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v31[2]; // [rsp+30h] [rbp-F0h] BYREF
  char v32; // [rsp+40h] [rbp-E0h] BYREF
  char *v33; // [rsp+80h] [rbp-A0h]
  char v34; // [rsp+98h] [rbp-88h] BYREF
  char *v35; // [rsp+A0h] [rbp-80h]
  char v36; // [rsp+B0h] [rbp-70h] BYREF
  char *v37; // [rsp+C0h] [rbp-60h]
  char v38; // [rsp+D0h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a1 + 960);
  v5 = *(_QWORD *)(v4 + 128);
  v6 = *(unsigned int *)(v4 + 144);
  if ( !(_DWORD)v6 )
    return 0;
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = v5 + 16LL * v8;
  v10 = *(int **)v9;
  if ( a2 != *(int **)v9 )
  {
    v24 = 1;
    while ( v10 != (int *)-4096LL )
    {
      v25 = v24 + 1;
      v8 = (v6 - 1) & (v24 + v8);
      v9 = v5 + 16LL * v8;
      v10 = *(int **)v9;
      if ( a2 == *(int **)v9 )
        goto LABEL_3;
      v24 = v25;
    }
    return 0;
  }
LABEL_3:
  if ( v9 == v5 + 16 * v6 )
    return 0;
  v11 = *(_QWORD *)(a1 + 864);
  v12 = *(_DWORD *)(v9 + 8);
  BYTE4(v29) = 0;
  v13 = sub_2E79000(*(__int64 **)(v11 + 40));
  sub_336FEE0(
    (__int64)v31,
    *(_QWORD *)(*(_QWORD *)(a1 + 864) + 64LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL),
    v13,
    v12,
    a3,
    v29);
  v14 = *(_QWORD *)(a1 + 864);
  v28 = 0;
  v15 = *(_DWORD *)(a1 + 848);
  v29 = 0;
  v27 = v14 + 288;
  v16 = *(_QWORD *)a1;
  v30 = v15;
  if ( v16 )
  {
    if ( &v29 != (__int64 *)(v16 + 48) )
    {
      v17 = *(_QWORD *)(v16 + 48);
      v29 = v17;
      if ( v17 )
      {
        sub_B96E90((__int64)&v29, v17, 1);
        v14 = *(_QWORD *)(a1 + 864);
      }
    }
  }
  v18 = sub_3370E50((__int64)v31, v14, *(_QWORD *)(a1 + 960), (__int64)&v29, (__int64)&v27, 0, a2);
  v19 = v18;
  v20 = v18;
  v22 = v21;
  if ( v29 )
  {
    v26 = v18;
    sub_B91220((__int64)&v29, v29);
    v19 = v26;
  }
  sub_3380540(a1, (__int64)a2, v19, v22);
  if ( v37 != &v38 )
    _libc_free((unsigned __int64)v37);
  if ( v35 != &v36 )
    _libc_free((unsigned __int64)v35);
  if ( v33 != &v34 )
    _libc_free((unsigned __int64)v33);
  if ( (char *)v31[0] != &v32 )
    _libc_free(v31[0]);
  return v20;
}
