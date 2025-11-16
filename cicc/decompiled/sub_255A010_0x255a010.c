// Function: sub_255A010
// Address: 0x255a010
//
__int64 __fastcall sub_255A010(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  unsigned __int64 v4; // rax
  __int64 v5; // rbx
  unsigned __int64 v6; // rdx
  unsigned int v7; // r14d
  char *v8; // r15
  unsigned __int8 v9; // al
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int16 v14; // ax
  int v15; // r15d
  unsigned __int8 *v16; // rax
  char v17; // al
  unsigned __int8 v18; // dl
  unsigned __int8 v19; // cl
  unsigned __int64 v20; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  unsigned __int16 v24; // di
  __int16 v25; // r8
  unsigned __int64 v26; // rax
  __int16 v27; // r8
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  __int64 *v30; // rdi
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  int v36; // eax
  int v37; // eax
  unsigned __int64 v38; // [rsp+0h] [rbp-80h]
  unsigned __int64 v39; // [rsp+0h] [rbp-80h]
  __int64 v41; // [rsp+18h] [rbp-68h] BYREF
  _BYTE *v42; // [rsp+20h] [rbp-60h] BYREF
  __int64 v43; // [rsp+28h] [rbp-58h]
  _BYTE v44[80]; // [rsp+30h] [rbp-50h] BYREF

  v2 = (__int64 *)(a1 + 72);
  v4 = sub_250D070((_QWORD *)(a1 + 72));
  v5 = *(_QWORD *)(v4 + 16);
  if ( v5 )
  {
    v6 = v4;
    v7 = 1;
    while ( 1 )
    {
      v8 = *(char **)(v5 + 24);
      v9 = *v8;
      if ( (unsigned __int8)*v8 <= 0x1Cu )
        goto LABEL_4;
      if ( v9 == 62 )
      {
        v10 = *((_QWORD *)v8 - 4);
        if ( !v10 )
          goto LABEL_4;
        if ( v6 != v10 )
          goto LABEL_4;
        v11 = *(_QWORD *)(a1 + 104);
        if ( v11 )
        {
          _BitScanReverse64(&v11, v11);
          _BitScanReverse64(&v12, 1LL << (*((_WORD *)v8 + 1) >> 1));
          if ( (unsigned __int8)(63 - (v12 ^ 0x3F)) >= (unsigned __int8)(63 - (v11 ^ 0x3F)) )
            goto LABEL_4;
        }
        if ( !byte_4FEF578 )
        {
          v38 = v6;
          v36 = sub_2207590((__int64)&byte_4FEF578);
          v6 = v38;
          if ( v36 )
          {
            sub_2207640((__int64)&byte_4FEF578);
            v6 = v38;
          }
        }
        v13 = *(_QWORD *)(a1 + 104);
        v14 = 510;
        if ( v13 )
        {
          _BitScanReverse64(&v13, v13);
          v14 = (2 * (63 - (v13 ^ 0x3F))) & 0x1FE;
        }
        v7 = 0;
        *((_WORD *)v8 + 1) = *((_WORD *)v8 + 1) & 0xFF81 | v14;
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          goto LABEL_15;
      }
      else
      {
        if ( v9 == 61 )
        {
          v22 = *((_QWORD *)v8 - 4);
          if ( v6 == v22 )
          {
            if ( v22 )
            {
              v23 = *(_QWORD *)(a1 + 104);
              v24 = *((_WORD *)v8 + 1);
              if ( !v23 )
              {
                v27 = 510;
                goto LABEL_37;
              }
              _BitScanReverse64(&v23, v23);
              v25 = 63 - (v23 ^ 0x3F);
              _BitScanReverse64(&v26, 1LL << (v24 >> 1));
              if ( (unsigned __int8)(63 - (v26 ^ 0x3F)) < (unsigned __int8)v25 )
              {
                v27 = (2 * v25) & 0x1FE;
LABEL_37:
                *((_WORD *)v8 + 1) = v27 | v24 & 0xFF81;
                if ( byte_4FEF570 || (v39 = v6, v37 = sub_2207590((__int64)&byte_4FEF570), v6 = v39, !v37) )
                {
                  v7 = 0;
                }
                else
                {
                  v7 = 0;
                  sub_2207640((__int64)&byte_4FEF570);
                  v6 = v39;
                }
              }
            }
          }
        }
LABEL_4:
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          goto LABEL_15;
      }
    }
  }
  v7 = 1;
LABEL_15:
  v15 = 1;
  if ( (unsigned int)*(unsigned __int8 *)sub_250D070(v2) - 12 > 1 )
  {
    v43 = 0x400000000LL;
    v28 = *(_QWORD *)(a1 + 72);
    v42 = v44;
    v29 = v28 & 0xFFFFFFFFFFFFFFFCLL;
    if ( (v28 & 3) == 3 )
      v29 = *(_QWORD *)(v29 + 24);
    v30 = (__int64 *)sub_BD5C60(v29);
    v31 = *(_QWORD *)(a1 + 104);
    if ( v31 )
    {
      _BitScanReverse64(&v31, v31);
      LODWORD(v31) = v31 ^ 0x3F;
      if ( (_DWORD)v31 != 63 )
      {
        v41 = sub_A77A40(v30, 63 - (unsigned __int8)v31);
        sub_25594F0((__int64)&v42, &v41, v32, v33, v34, v35);
      }
    }
    v15 = 1;
    if ( (_DWORD)v43 )
      v15 = sub_2516380(a2, v2, (__int64)v42, (unsigned int)v43, 0);
    if ( v42 != v44 )
      _libc_free((unsigned __int64)v42);
  }
  v16 = (unsigned __int8 *)sub_250D070(v2);
  v17 = sub_BD5420(v16, *(_QWORD *)(*(_QWORD *)(a2 + 208) + 104LL));
  v18 = -1;
  v19 = v17;
  v20 = *(_QWORD *)(a1 + 104);
  if ( v20 )
  {
    _BitScanReverse64(&v20, v20);
    v18 = 63 - (v20 ^ 0x3F);
  }
  if ( v19 < v18 )
    return sub_250C0B0(v15, v7);
  else
    return v7;
}
