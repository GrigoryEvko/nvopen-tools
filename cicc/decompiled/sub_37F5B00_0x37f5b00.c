// Function: sub_37F5B00
// Address: 0x37f5b00
//
unsigned __int64 __fastcall sub_37F5B00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r15d
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _BYTE *v14; // rdi
  unsigned int v15; // ecx
  __int16 *v16; // rsi
  int v17; // edx
  unsigned __int64 v18; // r14
  int v20; // eax
  __int64 v21; // rbx
  int v22; // r10d
  __int64 v23; // rcx
  unsigned int v24; // esi
  char v25; // al
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-98h]
  int v28; // [rsp+18h] [rbp-88h]
  __int64 v29; // [rsp+20h] [rbp-80h] BYREF
  _BYTE *v30; // [rsp+28h] [rbp-78h]
  __int64 v31; // [rsp+30h] [rbp-70h]
  _BYTE v32[48]; // [rsp+38h] [rbp-68h] BYREF
  int v33; // [rsp+68h] [rbp-38h]

  v6 = a3;
  v9 = *(_QWORD *)(a1 + 208);
  v30 = v32;
  v31 = 0x600000000LL;
  v29 = 0;
  v33 = 0;
  sub_2ED4FB0((__int64)&v29, v9, a3, a4, a5, a6);
  sub_2E225E0(&v29, a2, v10, v11, v12, v13);
  if ( v6 - 1 <= 0x3FFFFFFE )
  {
    v14 = v30;
    v15 = *(_DWORD *)(*(_QWORD *)(v29 + 8) + 24LL * v6 + 16) & 0xFFF;
    v16 = (__int16 *)(*(_QWORD *)(v29 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v29 + 8) + 24LL * v6 + 16) >> 12));
    while ( v16 )
    {
      if ( (*(_QWORD *)&v30[8 * (v15 >> 6)] & (1LL << v15)) != 0 )
      {
        v18 = sub_2E31A10(a2, 1);
        if ( v18 != a2 + 48 )
          goto LABEL_13;
        goto LABEL_28;
      }
      v17 = *v16++;
      v15 += v17;
      if ( !(_WORD)v17 )
        goto LABEL_6;
    }
    goto LABEL_6;
  }
  v18 = sub_2E31A10(a2, 1);
  if ( v18 == a2 + 48 )
  {
LABEL_28:
    v14 = v30;
LABEL_6:
    v18 = 0;
    goto LABEL_7;
  }
  if ( v6 - 0x40000000 <= 0x3FFFFFFF && sub_37F43B0(v18, v6 - 0x40000000, *(__int64 **)(a1 + 216)) )
  {
LABEL_22:
    v14 = v30;
    goto LABEL_7;
  }
LABEL_13:
  v20 = sub_37F56A0(a1, v18, v6);
  v21 = *(_QWORD *)(v18 + 32);
  v22 = v20;
  v23 = v21 + 40LL * (*(_DWORD *)(v18 + 40) & 0xFFFFFF);
  if ( v21 != v23 )
  {
    do
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v21 )
        {
          v24 = *(_DWORD *)(v21 + 8);
          if ( v24 )
          {
            if ( (*(_BYTE *)(v21 + 3) & 0x10) != 0 )
            {
              if ( v6 == v24 )
                goto LABEL_22;
              if ( v24 - 1 <= 0x3FFFFFFE && v6 - 1 <= 0x3FFFFFFE )
                break;
            }
          }
        }
        v21 += 40;
        if ( v23 == v21 )
          goto LABEL_25;
      }
      v27 = v23;
      v28 = v22;
      v25 = sub_E92070(*(_QWORD *)(a1 + 208), v24, v6);
      v22 = v28;
      v23 = v27;
      if ( v25 )
        goto LABEL_22;
      v21 += 40;
    }
    while ( v27 != v21 );
  }
LABEL_25:
  if ( v22 < 0 )
    goto LABEL_28;
  v26 = sub_37F5910(a1, a2, v22);
  v14 = v30;
  v18 = v26;
LABEL_7:
  if ( v14 != v32 )
    _libc_free((unsigned __int64)v14);
  return v18;
}
