// Function: sub_19A0320
// Address: 0x19a0320
//
__int64 __fastcall sub_19A0320(__int64 a1, __int64 a2, __int64 *a3)
{
  int v4; // r14d
  __int64 v6; // r15
  _QWORD *v7; // rdi
  __int64 v8; // rax
  int v9; // eax
  __int64 v10; // r8
  __int64 *v11; // r9
  __int64 v12; // r10
  int v13; // r11d
  unsigned int v14; // ecx
  size_t v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rbx
  bool v18; // al
  int v19; // eax
  __int64 *v20; // [rsp+0h] [rbp-C0h]
  __int64 *v21; // [rsp+0h] [rbp-C0h]
  size_t v22; // [rsp+8h] [rbp-B8h]
  size_t v23; // [rsp+8h] [rbp-B8h]
  __int64 v24; // [rsp+10h] [rbp-B0h]
  __int64 v25; // [rsp+10h] [rbp-B0h]
  int v26; // [rsp+18h] [rbp-A8h]
  int v27; // [rsp+18h] [rbp-A8h]
  unsigned int v28; // [rsp+1Ch] [rbp-A4h]
  unsigned int v29; // [rsp+1Ch] [rbp-A4h]
  __int64 v30; // [rsp+20h] [rbp-A0h]
  __int64 v31; // [rsp+20h] [rbp-A0h]
  _QWORD v32[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v33; // [rsp+70h] [rbp-50h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = *(_QWORD **)a2;
    v32[1] = 0x400000001LL;
    v8 = *(unsigned int *)(a2 + 8);
    v32[0] = &v33;
    v33 = -2;
    v9 = sub_199F760(v7, (__int64)&v7[v8]);
    v10 = *(unsigned int *)(a2 + 8);
    v11 = a3;
    v12 = 0;
    v13 = 1;
    v14 = (v4 - 1) & v9;
    v15 = 8 * v10;
    while ( 1 )
    {
      v16 = v6 + 48LL * v14;
      v17 = *(unsigned int *)(v16 + 8);
      if ( v10 == v17 )
      {
        v25 = v10;
        v27 = v13;
        v29 = v14;
        v31 = v12;
        if ( !v15 )
          goto LABEL_12;
        v21 = v11;
        v23 = v15;
        v19 = memcmp(*(const void **)a2, *(const void **)v16, v15);
        v15 = v23;
        v11 = v21;
        v12 = v31;
        v14 = v29;
        v13 = v27;
        v10 = v25;
        if ( !v19 )
        {
LABEL_12:
          *v11 = v16;
          return 1;
        }
      }
      if ( v17 == 1 && **(_QWORD **)v16 == -1 )
        break;
      v20 = v11;
      v22 = v15;
      v24 = v10;
      v26 = v13;
      v28 = v14;
      v30 = v12;
      v18 = sub_199CBA0(v16, (__int64)v32);
      v12 = v30;
      v10 = v24;
      v15 = v22;
      v11 = v20;
      if ( !v30 && v18 )
        v12 = v16;
      v13 = v26 + 1;
      v14 = (v4 - 1) & (v26 + v28);
    }
    if ( !v12 )
      v12 = v16;
    *v11 = v12;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
