// Function: sub_199FE30
// Address: 0x199fe30
//
__int64 __fastcall sub_199FE30(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v6; // r15
  _QWORD *v7; // rdi
  __int64 v8; // rax
  int v9; // eax
  __int64 v10; // r8
  _QWORD *v11; // r9
  __int64 v12; // r10
  int v13; // r11d
  int v14; // ecx
  unsigned int v15; // r14d
  size_t v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // r13
  bool v19; // al
  int v20; // eax
  _QWORD *v21; // [rsp+0h] [rbp-C0h]
  _QWORD *v22; // [rsp+0h] [rbp-C0h]
  size_t v23; // [rsp+8h] [rbp-B8h]
  size_t v24; // [rsp+8h] [rbp-B8h]
  __int64 v25; // [rsp+10h] [rbp-B0h]
  __int64 v26; // [rsp+10h] [rbp-B0h]
  __int64 v27; // [rsp+18h] [rbp-A8h]
  __int64 v28; // [rsp+18h] [rbp-A8h]
  int v29; // [rsp+20h] [rbp-A0h]
  int v30; // [rsp+20h] [rbp-A0h]
  int v31; // [rsp+24h] [rbp-9Ch]
  int v32; // [rsp+24h] [rbp-9Ch]
  _QWORD v33[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v34; // [rsp+70h] [rbp-50h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = *(_QWORD **)a2;
    v33[1] = 0x400000001LL;
    v8 = *(unsigned int *)(a2 + 8);
    v33[0] = &v34;
    v34 = -2;
    v9 = sub_199F760(v7, (__int64)&v7[v8]);
    v10 = *(unsigned int *)(a2 + 8);
    v11 = a3;
    v12 = 0;
    v13 = 1;
    v14 = v4 - 1;
    v15 = (v4 - 1) & v9;
    v16 = 8 * v10;
    while ( 1 )
    {
      v17 = v6 + 56LL * v15;
      v18 = *(unsigned int *)(v17 + 8);
      if ( v10 == v18 )
      {
        v26 = v10;
        v30 = v13;
        v28 = v12;
        v32 = v14;
        if ( !v16 )
          goto LABEL_12;
        v22 = v11;
        v24 = v16;
        v20 = memcmp(*(const void **)a2, *(const void **)v17, v16);
        v16 = v24;
        v11 = v22;
        v14 = v32;
        v12 = v28;
        v13 = v30;
        v10 = v26;
        if ( !v20 )
        {
LABEL_12:
          *v11 = v17;
          return 1;
        }
      }
      if ( v18 == 1 && **(_QWORD **)v17 == -1 )
        break;
      v21 = v11;
      v23 = v16;
      v25 = v10;
      v29 = v13;
      v27 = v12;
      v31 = v14;
      v19 = sub_199CBA0(v6 + 56LL * v15, (__int64)v33);
      v12 = v27;
      v14 = v31;
      v10 = v25;
      v16 = v23;
      v11 = v21;
      if ( !v27 && v19 )
        v12 = v6 + 56LL * v15;
      v13 = v29 + 1;
      v15 = v31 & (v29 + v15);
    }
    if ( !v12 )
      v12 = v6 + 56LL * v15;
    *v11 = v12;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
