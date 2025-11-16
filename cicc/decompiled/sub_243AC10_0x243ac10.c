// Function: sub_243AC10
// Address: 0x243ac10
//
void __fastcall sub_243AC10(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  _BYTE *v6; // rax
  __int64 **v7; // r11
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  char v10; // dl
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 **v17; // [rsp+18h] [rbp-108h]
  int v18; // [rsp+28h] [rbp-F8h]
  _BYTE v19[32]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v20; // [rsp+50h] [rbp-D0h]
  unsigned int *v21[2]; // [rsp+60h] [rbp-C0h] BYREF
  _BYTE v22[32]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v23; // [rsp+90h] [rbp-90h]
  __int64 v24; // [rsp+98h] [rbp-88h]
  __int16 v25; // [rsp+A0h] [rbp-80h]
  __int64 v26; // [rsp+A8h] [rbp-78h]
  void **v27; // [rsp+B0h] [rbp-70h]
  void **v28; // [rsp+B8h] [rbp-68h]
  __int64 v29; // [rsp+C0h] [rbp-60h]
  int v30; // [rsp+C8h] [rbp-58h]
  __int16 v31; // [rsp+CCh] [rbp-54h]
  char v32; // [rsp+CEh] [rbp-52h]
  __int64 v33; // [rsp+D0h] [rbp-50h]
  __int64 v34; // [rsp+D8h] [rbp-48h]
  void *v35; // [rsp+E0h] [rbp-40h] BYREF
  void *v36; // [rsp+E8h] [rbp-38h] BYREF

  v4 = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)v4 > 0x27 || (v16 = 0x8020000038LL, !_bittest64(&v16, v4)) )
  {
    v26 = sub_BD5C60(a2);
    v27 = &v35;
    v31 = 512;
    v21[0] = (unsigned int *)v22;
    v35 = &unk_49DA100;
    v28 = &v36;
    v21[1] = (unsigned int *)0x200000000LL;
    v25 = 0;
    v36 = &unk_49DA0B0;
    v29 = 0;
    v30 = 0;
    v32 = 7;
    v33 = 0;
    v34 = 0;
    v23 = 0;
    v24 = 0;
    sub_D5F1F0((__int64)v21, a2);
    v5 = *(_QWORD *)(a1 + 120);
    v20 = 257;
    v6 = sub_94BCF0(v21, a3, v5, (__int64)v19);
    v7 = *(__int64 ***)(a3 + 8);
    v20 = 257;
    v17 = v7;
    v8 = sub_2435400(*(_BYTE *)(a1 + 160), *(_DWORD *)(a1 + 176), *(_QWORD *)(a1 + 184), (__int64 *)v21, (__int64)v6);
    v9 = sub_2436E50((__int64 *)v21, 0x30u, v8, v17, (__int64)v19, 0, v18, 0);
    v10 = *(_BYTE *)a2;
    v11 = 0;
    if ( *(_BYTE *)a2 != 61 )
    {
      v11 = 32;
      if ( v10 != 62 )
      {
        v11 = 0;
        if ( (unsigned __int8)(v10 - 65) > 1u )
          sub_C64ED0("Unexpected instruction", 1u);
      }
    }
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v12 = *(_QWORD *)(a2 - 8);
    else
      v12 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v13 = v11 + v12;
    if ( *(_QWORD *)v13 )
    {
      v14 = *(_QWORD *)(v13 + 8);
      **(_QWORD **)(v13 + 16) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = *(_QWORD *)(v13 + 16);
    }
    *(_QWORD *)v13 = v9;
    if ( v9 )
    {
      v15 = *(_QWORD *)(v9 + 16);
      *(_QWORD *)(v13 + 8) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = v13 + 8;
      *(_QWORD *)(v13 + 16) = v9 + 16;
      *(_QWORD *)(v9 + 16) = v13;
    }
    nullsub_61();
    v35 = &unk_49DA100;
    nullsub_63();
    if ( (_BYTE *)v21[0] != v22 )
      _libc_free((unsigned __int64)v21[0]);
  }
}
