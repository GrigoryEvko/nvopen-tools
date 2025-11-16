// Function: sub_382D870
// Address: 0x382d870
//
__int64 __fastcall sub_382D870(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rax
  unsigned int v5; // r8d
  __int64 v6; // rcx
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r13
  unsigned __int16 v12; // ax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // r9d
  int v17; // edi
  bool v18; // al
  _QWORD *v19; // rdi
  __int64 v20; // r12
  unsigned int v22; // r8d
  unsigned __int16 v23; // r9
  __int64 v24; // r10
  _BYTE *v25; // r10
  int v26; // r11d
  _BYTE *v27; // rax
  int v28; // r9d
  unsigned int v29; // r14d
  unsigned int v30; // eax
  int v31; // r9d
  __int64 v32; // rax
  _QWORD *v33; // r14
  __int128 v34; // rax
  __int64 v35; // r9
  unsigned __int8 *v36; // rax
  unsigned int v37; // edx
  __int128 v38; // rax
  __int64 v39; // r9
  __int128 v40; // [rsp-50h] [rbp-F0h]
  __int128 v41; // [rsp-30h] [rbp-D0h]
  __int128 v42; // [rsp-30h] [rbp-D0h]
  unsigned int v43; // [rsp+4h] [rbp-9Ch]
  unsigned int v44; // [rsp+4h] [rbp-9Ch]
  __int64 v45; // [rsp+8h] [rbp-98h]
  int v46; // [rsp+8h] [rbp-98h]
  __int64 v47; // [rsp+8h] [rbp-98h]
  int v48; // [rsp+8h] [rbp-98h]
  unsigned __int16 v50; // [rsp+30h] [rbp-70h] BYREF
  __int64 v51; // [rsp+38h] [rbp-68h]
  unsigned int v52; // [rsp+40h] [rbp-60h] BYREF
  __int64 v53; // [rsp+48h] [rbp-58h]
  __int64 v54; // [rsp+50h] [rbp-50h] BYREF
  int v55; // [rsp+58h] [rbp-48h]
  unsigned __int64 v56; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v57; // [rsp+68h] [rbp-38h]

  v4 = sub_37AE0F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD *)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = v4;
  v9 = v4;
  v11 = v10;
  v12 = *(_WORD *)v6;
  v51 = *(_QWORD *)(v6 + 8);
  v13 = (unsigned int)v10;
  v50 = v12;
  v14 = *(_QWORD *)(v8 + 48) + 16LL * (unsigned int)v10;
  v45 = v13;
  LOWORD(v13) = *(_WORD *)v14;
  v15 = *(_QWORD *)(v14 + 8);
  v54 = v7;
  LOWORD(v52) = v13;
  v53 = v15;
  if ( v7 )
  {
    sub_B96E90((__int64)&v54, v7, 1);
    v12 = v50;
  }
  v55 = *(_DWORD *)(a2 + 72);
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 17) <= 0xD3u )
      goto LABEL_5;
  }
  else if ( sub_30070B0((__int64)&v50) )
  {
    goto LABEL_5;
  }
  if ( (_WORD)v52 )
  {
    if ( *(_QWORD *)(*a1 + 8LL * (unsigned __int16)v52 + 112) )
    {
      v7 = 198;
      if ( !(unsigned __int8)sub_38138F0(*a1, 0xC6u, v52, 0, v5) )
      {
        v7 = 203;
        if ( !(unsigned __int8)sub_38138F0(v24, 0xCBu, v23, 0, v22) )
        {
          v27 = &v25[500 * v26];
          if ( v27[6614] )
          {
            if ( v27[6613] )
            {
              v7 = a2;
              if ( sub_3459160(v25, a2, a1[1]) )
              {
                v20 = (__int64)sub_33FAF80(a1[1], 215, (__int64)&v54, v52, v53, v28, a3);
                goto LABEL_9;
              }
            }
          }
        }
      }
    }
  }
LABEL_5:
  v16 = *(_DWORD *)(a2 + 24);
  if ( v16 == 198 || (v17 = *(_DWORD *)(a2 + 24), v16 == 418) )
  {
    v46 = *(_DWORD *)(a2 + 24);
    v29 = sub_32844A0(&v50, v7);
    v30 = sub_32844A0((unsigned __int16 *)&v52, v7);
    v31 = v46;
    v57 = v30;
    if ( v30 > 0x40 )
    {
      sub_C43690((__int64)&v56, 0, 0);
      v32 = 1LL << v29;
      v31 = v46;
      if ( v57 > 0x40 )
      {
        *(_QWORD *)(v56 + 8LL * (v29 >> 6)) |= v32;
        goto LABEL_25;
      }
    }
    else
    {
      v56 = 0;
      v32 = 1LL << v29;
    }
    v56 |= v32;
LABEL_25:
    v33 = (_QWORD *)a1[1];
    if ( v31 == 198 )
    {
      *(_QWORD *)&v38 = sub_34007B0((__int64)v33, (__int64)&v56, (__int64)&v54, v52, v53, 0, a3, 0);
      *((_QWORD *)&v42 + 1) = v11;
      *(_QWORD *)&v42 = v8;
      v36 = sub_3406EB0(v33, 0xBBu, (__int64)&v54, v52, v53, v39, v42, v38);
      v16 = 203;
    }
    else
    {
      v47 = *(_QWORD *)(a2 + 40);
      *(_QWORD *)&v34 = sub_34007B0((__int64)v33, (__int64)&v56, (__int64)&v54, v52, v53, 0, a3, 0);
      *((_QWORD *)&v40 + 1) = v11;
      *(_QWORD *)&v40 = v8;
      v36 = sub_33FC130(v33, 400, (__int64)&v54, v52, v53, v35, v40, v34, *(_OWORD *)(v47 + 40), *(_OWORD *)(v47 + 80));
      v16 = 419;
    }
    v9 = (__int64)v36;
    if ( v57 > 0x40 && v56 )
    {
      v44 = v37;
      v48 = v16;
      j_j___libc_free_0_0(v56);
      v37 = v44;
      v16 = v48;
    }
    v17 = *(_DWORD *)(a2 + 24);
    v45 = v37;
  }
  v43 = v16;
  v18 = sub_33CB110(v17);
  v19 = (_QWORD *)a1[1];
  if ( v18 )
  {
    *((_QWORD *)&v41 + 1) = v11 & 0xFFFFFFFF00000000LL | v45;
    *(_QWORD *)&v41 = v9;
    v20 = sub_340F900(
            v19,
            v43,
            (__int64)&v54,
            v52,
            v53,
            v43,
            v41,
            *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
            *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  }
  else
  {
    v20 = (__int64)sub_33FAF80((__int64)v19, v43, (__int64)&v54, v52, v53, v43, a3);
  }
LABEL_9:
  if ( v54 )
    sub_B91220((__int64)&v54, v54);
  return v20;
}
