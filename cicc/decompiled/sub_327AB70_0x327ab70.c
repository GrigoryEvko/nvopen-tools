// Function: sub_327AB70
// Address: 0x327ab70
//
__int64 __fastcall sub_327AB70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        char a8)
{
  __int64 *v10; // rdi
  char v11; // bl
  unsigned int v13; // ebx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int i; // r15d
  unsigned __int16 v20; // ax
  __int64 v21; // rdx
  __int64 *v22; // r14
  unsigned int v23; // r9d
  __int64 v24; // rsi
  unsigned int v25; // eax
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // esi
  unsigned __int16 v31; // ax
  __int64 v32; // rdx
  __int64 v33; // rdx
  unsigned int v34; // eax
  unsigned int v35; // [rsp+8h] [rbp-88h]
  int v36; // [rsp+Ch] [rbp-84h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  unsigned int v39; // [rsp+24h] [rbp-6Ch]
  __int16 v40; // [rsp+24h] [rbp-6Ch]
  unsigned int v41; // [rsp+28h] [rbp-68h]
  __int64 v42; // [rsp+28h] [rbp-68h]
  __int64 v43; // [rsp+30h] [rbp-60h] BYREF
  __int64 v44; // [rsp+38h] [rbp-58h]
  unsigned int v45; // [rsp+40h] [rbp-50h] BYREF
  __int64 v46; // [rsp+48h] [rbp-48h]
  __int64 v47; // [rsp+50h] [rbp-40h]
  __int64 v48; // [rsp+58h] [rbp-38h]

  v10 = *(__int64 **)(a6 + 40);
  v43 = a3;
  v35 = a2;
  v44 = a4;
  v11 = *(_BYTE *)sub_2E79000(v10);
  if ( !(_WORD)v43 )
  {
    if ( !sub_3007070((__int64)&v43) || v11 )
      goto LABEL_5;
    if ( !sub_3007100((__int64)&v43) )
      goto LABEL_10;
    goto LABEL_49;
  }
  if ( (unsigned __int16)(v43 - 2) > 7u && (unsigned __int16)(v43 - 17) > 0x6Cu && (unsigned __int16)(v43 - 176) > 0x1Fu
    || v11 )
  {
    goto LABEL_5;
  }
  if ( (unsigned __int16)(v43 - 176) <= 0x34u )
  {
LABEL_49:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    v31 = v43;
    if ( (_WORD)v43 )
    {
      if ( (unsigned __int16)(v43 - 176) > 0x34u )
      {
        v32 = (unsigned __int16)v43 - 1;
        v13 = word_4456340[v32];
LABEL_37:
        if ( (unsigned __int16)(v31 - 17) <= 0xD3u )
        {
          v31 = word_4456580[v32];
          v33 = 0;
        }
        else
        {
          v33 = v44;
        }
        goto LABEL_39;
      }
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
      goto LABEL_36;
    }
LABEL_10:
    v13 = sub_3007130((__int64)&v43, a2);
    goto LABEL_11;
  }
LABEL_36:
  v31 = v43;
  v32 = (unsigned __int16)v43 - 1;
  v13 = word_4456340[v32];
  if ( (_WORD)v43 )
    goto LABEL_37;
LABEL_11:
  if ( !sub_30070B0((__int64)&v43) )
  {
    LOWORD(v45) = 0;
    v46 = v44;
LABEL_13:
    v17 = sub_3007260((__int64)&v45);
    v47 = v17;
    v48 = v18;
    goto LABEL_14;
  }
  v31 = sub_3009970((__int64)&v43, a2, v14, v15, v16);
LABEL_39:
  LOWORD(v45) = v31;
  v46 = v33;
  if ( !v31 )
    goto LABEL_13;
  if ( v31 == 1 || (unsigned __int16)(v31 - 504) <= 7u )
    BUG();
  v17 = *(_QWORD *)&byte_444C4A0[16 * v31 - 16];
LABEL_14:
  v36 = v17;
  for ( i = 2; i < v13; i *= 2 )
  {
    v23 = v13 / i;
    if ( !(v13 % i) )
    {
      v22 = *(__int64 **)(a6 + 64);
      v30 = i * v36;
      if ( i * v36 == 2 )
      {
        v20 = 3;
        v21 = 0;
      }
      else
      {
        switch ( v30 )
        {
          case 4u:
            v20 = 4;
            v21 = 0;
            break;
          case 8u:
            v20 = 5;
            v21 = 0;
            break;
          case 0x10u:
            v20 = 6;
            v21 = 0;
            break;
          case 0x20u:
            v20 = 7;
            v21 = 0;
            break;
          case 0x40u:
            v20 = 8;
            v21 = 0;
            break;
          case 0x80u:
            v20 = 9;
            v21 = 0;
            break;
          default:
            v20 = sub_3007020(*(_QWORD **)(a6 + 64), v30);
            v22 = *(__int64 **)(a6 + 64);
            v23 = v13 / i;
            break;
        }
      }
      v24 = v23;
      v38 = v21;
      v39 = v23;
      v41 = v20;
      v25 = sub_2D43050(v20, v23);
      v26 = v39;
      v27 = v25;
      if ( (_WORD)v25 )
      {
        v28 = 0;
      }
      else
      {
        v24 = v41;
        v34 = sub_3009400(v22, v41, v38, v39, 0);
        v27 = v34;
        if ( !(_WORD)v34 )
          continue;
      }
      v29 = (unsigned __int16)v27;
      if ( *(_QWORD *)(a7 + 8LL * (unsigned __int16)v27 + 112) )
      {
        if ( !a8 || (v29 = v35, (*(_BYTE *)(v35 + a7 + 500LL * (unsigned __int16)v27 + 6414) & 0xFB) == 0) )
        {
          v45 = i;
          if ( !*(_QWORD *)(a5 + 16) )
            sub_4263D6(a7, v24, v28);
          v40 = v27;
          v42 = v28;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned int *, __int64, __int64, __int64, __int64))(a5 + 24))(
                 a5,
                 &v45,
                 v28,
                 v29,
                 v27,
                 v26) )
          {
            *(_WORD *)a1 = v40;
            *(_QWORD *)(a1 + 8) = v42;
            *(_BYTE *)(a1 + 16) = 1;
            return a1;
          }
        }
      }
    }
  }
LABEL_5:
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
