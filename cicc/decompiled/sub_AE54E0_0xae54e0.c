// Function: sub_AE54E0
// Address: 0xae54e0
//
__int64 __fastcall sub_AE54E0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  unsigned __int64 v4; // rsi
  __int64 v5; // rbx
  _QWORD *v6; // r15
  __int64 v7; // r12
  unsigned __int64 v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  unsigned __int64 v11; // r10
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int8 v15; // dl
  unsigned __int64 v16; // r10
  __int64 v18; // r11
  __int64 v19; // rbx
  __int64 v20; // rdx
  unsigned __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // r10
  __int64 v25; // r11
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  bool v28; // zf
  unsigned __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned __int64 v35; // [rsp+0h] [rbp-70h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  unsigned __int64 v37; // [rsp+10h] [rbp-60h]
  char v38; // [rsp+18h] [rbp-58h]
  unsigned __int64 v39; // [rsp+18h] [rbp-58h]
  __int64 v40; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+18h] [rbp-58h]
  __int64 v42; // [rsp+18h] [rbp-58h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  _QWORD *v44; // [rsp+28h] [rbp-48h]
  unsigned __int64 v45; // [rsp+30h] [rbp-40h] BYREF
  __int64 v46; // [rsp+38h] [rbp-38h]

  v4 = a2 & 0xFFFFFFFFFFFFFFF9LL | 4;
  v44 = &a3[a4];
  if ( v44 != a3 )
  {
    v5 = v4;
    v6 = a3;
    v7 = 0;
    while ( 1 )
    {
      v8 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      v9 = *(_DWORD *)(*v6 + 32LL);
      v10 = *(__int64 **)(*v6 + 24LL);
      v11 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v5 && (v5 & 6) == 0 && v8 )
      {
        v12 = *(_QWORD *)(*v6 + 24LL);
        if ( v9 > 0x40 )
          v12 = *v10;
        v13 = 16LL * (unsigned int)v12 + sub_AE4AC0(a1, v8) + 24;
        v14 = *(_QWORD *)v13;
        LOBYTE(v13) = *(_BYTE *)(v13 + 8);
        v45 = v14;
        LOBYTE(v46) = v13;
        v7 += sub_CA1930(&v45);
        goto LABEL_11;
      }
      if ( v9 > 0x40 )
        break;
      if ( v9 )
      {
        v18 = (__int64)((_QWORD)v10 << (64 - (unsigned __int8)v9)) >> (64 - (unsigned __int8)v9);
        if ( v18 )
          goto LABEL_24;
      }
LABEL_19:
      if ( v5 )
      {
        v19 = (v5 >> 1) & 3;
        if ( v19 == 2 )
        {
          if ( v8 )
            goto LABEL_12;
        }
        else if ( v19 == 1 && v8 )
        {
          v11 = *(_QWORD *)(v8 + 24);
          goto LABEL_12;
        }
      }
LABEL_11:
      v11 = sub_BCBAE0(v8, *v6);
LABEL_12:
      v15 = *(_BYTE *)(v11 + 8);
      if ( v15 == 16 )
      {
        v5 = *(_QWORD *)(v11 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
LABEL_4:
        if ( v44 == ++v6 )
          return v7;
      }
      else
      {
        v16 = v11 & 0xFFFFFFFFFFFFFFF9LL;
        if ( (unsigned int)v15 - 17 > 1 )
        {
          v28 = v15 == 15;
          v29 = 0;
          if ( v28 )
            v29 = v16;
          v5 = v29;
          goto LABEL_4;
        }
        ++v6;
        v5 = v16 | 2;
        if ( v44 == v6 )
          return v7;
      }
    }
    v18 = *v10;
    if ( !*v10 )
      goto LABEL_19;
LABEL_24:
    v20 = (v5 >> 1) & 3;
    if ( !v5 )
    {
      v42 = v18;
      v33 = sub_BCBAE0(v8, *v6);
      v18 = v42;
      v11 = 0;
      v21 = v33;
      goto LABEL_27;
    }
    if ( v20 == 2 )
    {
      v21 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 )
      {
LABEL_27:
        v35 = v11;
        v36 = v18;
        v38 = sub_AE5020(a1, v21);
        v22 = sub_9208B0(a1, v21);
        v24 = v35;
        v46 = v23;
        v25 = v36;
        v26 = ((1LL << v38) + ((unsigned __int64)(v22 + 7) >> 3) - 1) >> v38 << v38;
LABEL_28:
        v39 = v24;
        LOBYTE(v46) = v23;
        v45 = v26 * v25;
        v27 = sub_CA1930(&v45);
        v11 = v39;
        v7 += v27;
        goto LABEL_19;
      }
    }
    else if ( v20 == 1 )
    {
      if ( v8 )
      {
        v30 = *(_QWORD *)(v8 + 24);
      }
      else
      {
        v43 = v18;
        v34 = sub_BCBAE0(0, *v6);
        v18 = v43;
        v11 = v5 & 0xFFFFFFFFFFFFFFF8LL;
        v30 = v34;
      }
      v37 = v11;
      v40 = v18;
      v31 = sub_9208B0(a1, v30);
      v25 = v40;
      v24 = v37;
      v46 = v23;
      v26 = (unsigned __int64)(v31 + 7) >> 3;
      goto LABEL_28;
    }
    v41 = v18;
    v32 = sub_BCBAE0(v5 & 0xFFFFFFFFFFFFFFF8LL, *v6);
    v18 = v41;
    v11 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    v21 = v32;
    goto LABEL_27;
  }
  return 0;
}
