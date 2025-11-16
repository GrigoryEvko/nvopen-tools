// Function: sub_33ED970
// Address: 0x33ed970
//
_QWORD *__fastcall sub_33ED970(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  size_t v4; // rdx
  const char *v7; // r15
  __int64 v8; // r13
  _BYTE *v9; // rax
  unsigned __int64 v10; // r13
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int); // r15
  unsigned int v12; // esi
  __int64 v13; // rax
  int v14; // edx
  unsigned __int16 v15; // ax
  void *v16; // rsi
  unsigned int v17; // r15d
  _QWORD *v18; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-C8h]
  _QWORD v23[2]; // [rsp+10h] [rbp-C0h] BYREF
  char v24; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD *v25; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v26; // [rsp+50h] [rbp-80h]
  void *v27; // [rsp+60h] [rbp-70h] BYREF
  __int64 v28; // [rsp+68h] [rbp-68h]
  __int64 v29; // [rsp+70h] [rbp-60h]
  __int64 v30; // [rsp+78h] [rbp-58h]
  __int64 v31; // [rsp+80h] [rbp-50h]
  __int64 v32; // [rsp+88h] [rbp-48h]
  _QWORD *v33; // [rsp+90h] [rbp-40h]

  v4 = 0;
  v7 = *(const char **)(a2 + 96);
  v8 = *(_QWORD *)(**(_QWORD **)(a1 + 40) + 40LL);
  if ( v7 )
    v4 = strlen(*(const char **)(a2 + 96));
  v9 = sub_BA8CB0(v8, (__int64)v7, v4);
  v10 = (unsigned __int64)v9;
  if ( a4 )
    *a4 = v9;
  if ( !v9 )
  {
    v23[0] = &v24;
    v23[1] = 0;
    v32 = 0x100000000LL;
    v24 = 0;
    v28 = 0;
    v27 = &unk_49DD210;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v33 = v23;
    sub_CB5980((__int64)&v27, 0, 0, 0);
    sub_904010((__int64)&v27, "Undefined external symbol ");
    v20 = sub_A51310((__int64)&v27, 0x22u);
    v21 = sub_904010(v20, v7);
    sub_A51310(v21, 0x22u);
    v26 = 260;
    v25 = v23;
    sub_C64D30((__int64)&v25, 1u);
  }
  v22 = *(_QWORD *)(a1 + 16);
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v22 + 32LL);
  v12 = *(_DWORD *)(*((_QWORD *)v9 + 1) + 8LL) >> 8;
  v13 = sub_2E79000(*(__int64 **)(a1 + 40));
  if ( v11 == sub_2D42F30 )
  {
    v14 = sub_AE2980(v13, v12)[1];
    v15 = 2;
    if ( v14 != 1 )
    {
      v15 = 3;
      if ( v14 != 2 )
      {
        v15 = 4;
        if ( v14 != 4 )
        {
          v15 = 5;
          if ( v14 != 8 )
          {
            v15 = 6;
            if ( v14 != 16 )
            {
              v15 = 7;
              if ( v14 != 32 )
              {
                v15 = 8;
                if ( v14 != 64 )
                  v15 = 9 * (v14 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v15 = v11(v22, v13, v12);
  }
  v16 = *(void **)(a2 + 80);
  v17 = v15;
  v27 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v27, (__int64)v16, 1);
  LODWORD(v28) = *(_DWORD *)(a2 + 72);
  v18 = sub_33ED290(a1, v10, (__int64)&v27, v17, 0, 0, 0, 0);
  if ( v27 )
    sub_B91220((__int64)&v27, (__int64)v27);
  return v18;
}
