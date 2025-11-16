// Function: sub_2472900
// Address: 0x2472900
//
__int64 *__fastcall sub_2472900(__int64 a1)
{
  __int64 *result; // rax
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rsi
  _QWORD *v6; // rdi
  unsigned __int64 v7; // rax
  unsigned __int16 v8; // ax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r15
  _QWORD *v12; // rax
  __int64 v13; // r9
  __int64 v14; // r12
  unsigned int *v15; // r15
  unsigned int *v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int16 v19; // dx
  unsigned __int8 v20; // bl
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rsi
  __int64 v25; // rdx
  _BYTE *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  _BYTE *v30; // r9
  char v31; // al
  __int64 v32; // rax
  unsigned int v33; // ecx
  __int64 v34; // rdx
  unsigned int v35; // eax
  unsigned int v36; // edx
  __int64 v37; // rdx
  __int64 v38; // rax
  _BYTE *v39; // rdx
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 *v42; // rsi
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int64 v47; // r13
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // rbx
  __int64 *v51; // rax
  __int64 *v52; // rax
  __int64 v53; // [rsp+8h] [rbp-208h]
  __int64 v54; // [rsp+10h] [rbp-200h]
  __int64 *v55; // [rsp+18h] [rbp-1F8h]
  unsigned __int8 v56; // [rsp+24h] [rbp-1ECh]
  __int64 v57; // [rsp+30h] [rbp-1E0h]
  __int64 v58; // [rsp+38h] [rbp-1D8h]
  unsigned __int8 v59; // [rsp+4Bh] [rbp-1C5h]
  __int64 v60; // [rsp+50h] [rbp-1C0h]
  __int64 v61; // [rsp+50h] [rbp-1C0h]
  _BYTE *v62; // [rsp+50h] [rbp-1C0h]
  __int64 *v63; // [rsp+58h] [rbp-1B8h]
  __int64 v64; // [rsp+68h] [rbp-1A8h] BYREF
  unsigned __int64 v65; // [rsp+70h] [rbp-1A0h]
  __int64 v66; // [rsp+78h] [rbp-198h]
  __int64 v67; // [rsp+80h] [rbp-190h]
  __int64 v68; // [rsp+88h] [rbp-188h]
  _QWORD v69[4]; // [rsp+90h] [rbp-180h] BYREF
  __int16 v70; // [rsp+B0h] [rbp-160h]
  unsigned int *v71; // [rsp+C0h] [rbp-150h] BYREF
  int v72; // [rsp+C8h] [rbp-148h]
  char v73; // [rsp+D0h] [rbp-140h] BYREF
  __int64 v74; // [rsp+F8h] [rbp-118h]
  __int64 v75; // [rsp+100h] [rbp-110h]
  _QWORD *v76; // [rsp+108h] [rbp-108h]
  __int64 v77; // [rsp+118h] [rbp-F8h]
  void *v78; // [rsp+140h] [rbp-D0h]
  __m128i v79; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v80; // [rsp+160h] [rbp-B0h]
  __int64 v81; // [rsp+168h] [rbp-A8h]
  __int64 v82; // [rsp+170h] [rbp-A0h]
  __int64 v83; // [rsp+178h] [rbp-98h]
  __int64 v84; // [rsp+180h] [rbp-90h]
  __int64 v85; // [rsp+188h] [rbp-88h]
  __int16 v86; // [rsp+190h] [rbp-80h]

  result = *(__int64 **)(a1 + 1528);
  v63 = result;
  v55 = &result[*(unsigned int *)(a1 + 1536)];
  if ( result != v55 )
  {
    do
    {
      v3 = *v63;
      sub_23D0AB0((__int64)&v71, *v63, 0, 0, 0);
      v4 = *(_QWORD *)(v3 - 64);
      v57 = v4;
      v58 = *(_QWORD *)(v3 - 32);
      if ( sub_B46500((unsigned __int8 *)v3) )
      {
        v5 = *(_QWORD *)(v4 + 8);
        v6 = sub_2463540((__int64 *)a1, v5);
        if ( !v6 )
          BUG();
        v60 = sub_AD6530((__int64)v6, v5);
      }
      else
      {
        v60 = sub_246F3F0(a1, v4);
      }
      _BitScanReverse64(&v7, 1LL << (*(_WORD *)(v3 + 2) >> 1));
      v56 = 63 - (v7 ^ 0x3F);
      v59 = byte_4FE8EA9;
      LOBYTE(v8) = v56;
      HIBYTE(v8) = 1;
      if ( **(_BYTE **)(a1 + 8) )
        v9 = (unsigned __int64)sub_2465B30((__int64 *)a1, v58, (__int64)&v71, *(_QWORD *)(v60 + 8), 1);
      else
        v9 = sub_2463FC0(a1, v58, &v71, v8);
      v54 = v10;
      v11 = v9;
      LOWORD(v82) = 257;
      v12 = sub_BD2C40(80, unk_3F10A10);
      v14 = (__int64)v12;
      if ( v12 )
        sub_B4D3C0((__int64)v12, v60, v11, 0, v56, v13, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v77 + 16LL))(
        v77,
        v14,
        &v79,
        v74,
        v75);
      v15 = v71;
      v16 = &v71[4 * v72];
      if ( v71 != v16 )
      {
        do
        {
          v17 = *((_QWORD *)v15 + 1);
          v18 = *v15;
          v15 += 4;
          sub_B99FD0(v14, v18, v17);
        }
        while ( v16 != v15 );
      }
      if ( sub_B46500((unsigned __int8 *)v3) )
      {
        switch ( (*(_WORD *)(v3 + 2) >> 7) & 7 )
        {
          case 0:
            v19 = 0;
            break;
          case 1:
          case 2:
          case 5:
            v19 = 640;
            break;
          case 3:
            BUG();
          case 4:
          case 6:
            v19 = 768;
            break;
          case 7:
            v19 = 896;
            break;
        }
        *(_WORD *)(v3 + 2) = v19 | *(_WORD *)(v3 + 2) & 0xFC7F;
        if ( !*(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) || sub_B46500((unsigned __int8 *)v3) )
          goto LABEL_13;
      }
      else if ( !*(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
      {
        goto LABEL_13;
      }
      v20 = v56;
      if ( v59 >= v56 )
        v20 = v59;
      v21 = sub_246EE10(a1, v57);
      v22 = sub_B2BEC0(*(_QWORD *)a1);
      v23 = v22;
      if ( (unsigned __int8)byte_4FE8EA9 >= v20 )
        v20 = byte_4FE8EA9;
      v24 = v60;
      v79.m128i_i64[0] = sub_9208B0(v22, *(_QWORD *)(v60 + 8));
      v79.m128i_i64[1] = v25;
      v65 = (unsigned __int64)(v79.m128i_i64[0] + 7) >> 3;
      LOBYTE(v66) = v25;
      v26 = (_BYTE *)sub_24650D0(a1, v60, (__int64)&v71);
      v30 = v26;
      if ( *v26 > 0x15u )
        goto LABEL_31;
      if ( (_BYTE)qword_4FE7EA8 )
      {
        v61 = (__int64)v26;
        if ( !sub_AD7890((__int64)v26, v24, v27, v28, v29) )
        {
          v79 = (__m128i)(unsigned __int64)v23;
          v86 = 257;
          v80 = 0;
          v81 = 0;
          v82 = 0;
          v83 = 0;
          v84 = 0;
          v85 = 0;
          v31 = sub_9B6260(v61, &v79, 0);
          v30 = (_BYTE *)v61;
          if ( v31 )
          {
            v46 = *(_QWORD *)(a1 + 8);
            v69[0] = v21;
            if ( *(int *)(v46 + 4) > 1 )
            {
              LOWORD(v82) = 257;
              v21 = sub_921880(&v71, *(_QWORD *)(v46 + 360), *(_QWORD *)(v46 + 368), (int)v69, 1, (__int64)&v79, 0);
            }
            sub_24677C0((__int64 *)a1, (__int64)&v71, v21, v54, v65, v66, v20);
            goto LABEL_13;
          }
LABEL_31:
          v62 = v30;
          v32 = sub_9208B0(v23, *((_QWORD *)v30 + 1));
          v33 = 4;
          v68 = v34;
          v67 = v32;
          if ( !(_BYTE)v34 )
          {
            v33 = 0;
            if ( (unsigned int)v32 > 8 )
            {
              v35 = ((unsigned int)(v32 + 7) >> 3) - 1;
              v33 = v35;
              if ( v35 )
              {
                _BitScanReverse(&v36, v35);
                v33 = 32 - (v36 ^ 0x1F);
              }
            }
          }
          if ( *v62 <= 0x15u
            || (v37 = (int)qword_4FE8148, v38 = *(_QWORD *)(a1 + 1672) + 1LL, *(_QWORD *)(a1 + 1672) = v38, (int)v37 < 0)
            || v33 > 3
            || v38 <= v37
            || (v39 = *(_BYTE **)(a1 + 8), *v39) )
          {
            v79.m128i_i64[0] = (__int64)"_mscmp";
            LOWORD(v82) = 259;
            v40 = sub_2465600(a1, (__int64)v62, (__int64)&v71, (__int64)&v79);
            v41 = v74;
            if ( v74 )
              v41 = v74 - 24;
            v42 = (__int64 *)(v41 + 24);
            v43 = v53;
            LOWORD(v43) = 0;
            v53 = v43;
            v44 = sub_F38250(v40, v42, v43, 0, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 736LL), 0, 0, 0);
            sub_23D0AB0((__int64)&v79, v44, 0, 0, 0);
            v45 = *(_QWORD *)(a1 + 8);
            v64 = v21;
            if ( *(int *)(v45 + 4) > 1 )
            {
              v70 = 257;
              v21 = sub_921880(
                      (unsigned int **)&v79,
                      *(_QWORD *)(v45 + 360),
                      *(_QWORD *)(v45 + 368),
                      (int)&v64,
                      1,
                      (__int64)v69,
                      0);
            }
            sub_24677C0((__int64 *)a1, (__int64)&v79, v21, v54, v65, v66, v20);
            sub_F94A20(&v79, (__int64)&v79);
          }
          else
          {
            v47 = *(_QWORD *)&v39[16 * v33 + 248];
            v48 = *(_QWORD *)&v39[16 * v33 + 256];
            LOWORD(v82) = 257;
            v49 = sub_BCD140(v76, 8 << v33);
            v69[0] = sub_A82F30(&v71, (__int64)v62, v49, (__int64)&v79, 0);
            LOWORD(v82) = 257;
            v69[2] = v21;
            v69[1] = v58;
            v50 = sub_921880(&v71, v47, v48, (int)v69, 3, (__int64)&v79, 0);
            v51 = (__int64 *)sub_BD5C60(v50);
            *(_QWORD *)(v50 + 72) = sub_A7A090((__int64 *)(v50 + 72), v51, 1, 79);
            v52 = (__int64 *)sub_BD5C60(v50);
            *(_QWORD *)(v50 + 72) = sub_A7A090((__int64 *)(v50 + 72), v52, 3, 79);
          }
        }
      }
LABEL_13:
      nullsub_61();
      v78 = &unk_49DA100;
      nullsub_63();
      if ( v71 != (unsigned int *)&v73 )
        _libc_free((unsigned __int64)v71);
      result = ++v63;
    }
    while ( v55 != v63 );
  }
  return result;
}
