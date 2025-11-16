// Function: sub_1BA3A10
// Address: 0x1ba3a10
//
__int64 __fastcall sub_1BA3A10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        double a9)
{
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int v17; // esi
  char v18; // di
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r12
  _QWORD *v24; // rbx
  int v25; // esi
  unsigned int v26; // edx
  __int64 *v27; // rax
  _QWORD *v28; // rdi
  __int64 v29; // r13
  _QWORD *v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // r12
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // r12
  bool v36; // zf
  __int64 v37; // rdi
  __int64 v38; // rax
  _QWORD *v39; // r12
  int v40; // esi
  unsigned int v41; // edx
  __int64 *v42; // rax
  __int64 *v43; // rdi
  unsigned __int64 v45; // rsi
  __int64 v46; // rax
  unsigned int v47; // eax
  __int64 *v48; // r12
  __int64 *v49; // rdx
  __int64 v50; // rax
  __int64 *v51; // rbx
  __int64 v52; // rdi
  unsigned int v53; // esi
  int v54; // edx
  __int64 v55; // rax
  int v56; // r10d
  int v57; // edi
  __int64 *v58; // r11
  unsigned int v59; // [rsp+Ch] [rbp-114h]
  unsigned __int64 *v60; // [rsp+10h] [rbp-110h]
  __int64 v61; // [rsp+10h] [rbp-110h]
  __int64 v63; // [rsp+28h] [rbp-F8h]
  __int64 v66[2]; // [rsp+40h] [rbp-E0h] BYREF
  __int16 v67; // [rsp+50h] [rbp-D0h]
  __int64 v68[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v69; // [rsp+70h] [rbp-B0h]
  __int64 v70; // [rsp+80h] [rbp-A0h] BYREF
  __int64 *v71; // [rsp+88h] [rbp-98h]
  __int64 v72; // [rsp+90h] [rbp-90h]
  unsigned int v73; // [rsp+98h] [rbp-88h]
  __int64 *v74; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v75; // [rsp+A8h] [rbp-78h]
  unsigned __int64 *v76; // [rsp+B0h] [rbp-70h]
  __int64 v77; // [rsp+B8h] [rbp-68h]
  __int64 v78; // [rsp+C0h] [rbp-60h]
  int v79; // [rsp+C8h] [rbp-58h]
  __int64 v80; // [rsp+D0h] [rbp-50h]
  __int64 v81; // [rsp+D8h] [rbp-48h]

  v12 = *(_QWORD *)(a1 + 8);
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v13 = sub_13FCB50(v12);
  v17 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
  {
    v15 = v13;
    v18 = *(_BYTE *)(a2 + 23) & 0x40;
    v19 = 24LL * *(unsigned int *)(a2 + 56) + 8;
    v20 = 0;
    do
    {
      v14 = a2 - 24LL * v17;
      if ( v18 )
        v14 = *(_QWORD *)(a2 - 8);
      if ( v15 == *(_QWORD *)(v14 + v19) )
      {
        v21 = 24 * v20;
        goto LABEL_8;
      }
      ++v20;
      v19 += 8;
    }
    while ( v17 != (_DWORD)v20 );
    v21 = 0x17FFFFFFE8LL;
  }
  else
  {
    v21 = 0x17FFFFFFE8LL;
    v18 = *(_BYTE *)(a2 + 23) & 0x40;
  }
LABEL_8:
  if ( v18 )
  {
    v22 = *(_QWORD *)(a2 - 8);
  }
  else
  {
    v14 = 24LL * v17;
    v22 = a2 - v14;
  }
  v23 = *(_QWORD *)(*(_QWORD *)(v22 + v21) + 8LL);
  if ( v23 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v24 = sub_1648700(v23);
        if ( !sub_1377F70(*(_QWORD *)(a1 + 8) + 56LL, v24[5]) )
          break;
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          goto LABEL_17;
      }
      v25 = v73;
      v68[0] = (__int64)v24;
      if ( !v73 )
        break;
      v16 = v73 - 1;
      v15 = (__int64)v71;
      v26 = v16 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v27 = &v71[2 * v26];
      v28 = (_QWORD *)*v27;
      if ( v24 != (_QWORD *)*v27 )
      {
        v56 = 1;
        v14 = 0;
        while ( v28 != (_QWORD *)-8LL )
        {
          if ( v28 == (_QWORD *)-16LL && !v14 )
            v14 = (__int64)v27;
          v26 = v16 & (v56 + v26);
          v27 = &v71[2 * v26];
          v28 = (_QWORD *)*v27;
          if ( v24 == (_QWORD *)*v27 )
            goto LABEL_16;
          ++v56;
        }
        if ( v14 )
          v27 = (__int64 *)v14;
        v14 = (unsigned int)v72;
        ++v70;
        v57 = v72 + 1;
        if ( 4 * ((int)v72 + 1) < 3 * v73 )
        {
          v15 = v73 >> 3;
          if ( v73 - HIDWORD(v72) - v57 > (unsigned int)v15 )
          {
LABEL_67:
            LODWORD(v72) = v57;
            if ( *v27 != -8 )
              --HIDWORD(v72);
            *v27 = (__int64)v24;
            v27[1] = 0;
            goto LABEL_16;
          }
LABEL_72:
          sub_176F940((__int64)&v70, v25);
          sub_176A9A0((__int64)&v70, v68, &v74);
          v14 = (unsigned int)v72;
          v27 = v74;
          v24 = (_QWORD *)v68[0];
          v57 = v72 + 1;
          goto LABEL_67;
        }
LABEL_71:
        v25 = 2 * v73;
        goto LABEL_72;
      }
LABEL_16:
      v27[1] = a5;
      v23 = *(_QWORD *)(v23 + 8);
      if ( !v23 )
        goto LABEL_17;
    }
    ++v70;
    goto LABEL_71;
  }
LABEL_17:
  v29 = *(_QWORD *)(a2 + 8);
  if ( v29 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v30 = sub_1648700(v29);
        if ( !sub_1377F70(*(_QWORD *)(a1 + 8) + 56LL, v30[5]) )
          break;
LABEL_19:
        v29 = *(_QWORD *)(v29 + 8);
        if ( !v29 )
          goto LABEL_30;
      }
      v31 = sub_157EB90(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 32LL));
      v63 = sub_1632FA0(v31);
      v32 = sub_157EBA0(a6);
      v33 = sub_16498A0(v32);
      v74 = 0;
      v77 = v33;
      v76 = 0;
      v78 = 0;
      v79 = 0;
      v80 = 0;
      v81 = 0;
      v75 = 0;
      sub_17050D0((__int64 *)&v74, v32);
      v67 = 257;
      v34 = sub_15A0680(*(_QWORD *)a4, 1, 0);
      if ( *(_BYTE *)(a4 + 16) > 0x10u || *(_BYTE *)(v34 + 16) > 0x10u )
      {
        v69 = 257;
        v35 = (__int64 *)sub_15FB440(13, (__int64 *)a4, v34, (__int64)v68, 0);
        if ( v75 )
        {
          v60 = v76;
          sub_157E9D0(v75 + 40, (__int64)v35);
          v45 = *v60;
          v46 = v35[3] & 7;
          v35[4] = (__int64)v60;
          v45 &= 0xFFFFFFFFFFFFFFF8LL;
          v35[3] = v45 | v46;
          *(_QWORD *)(v45 + 8) = v35 + 3;
          *v60 = *v60 & 7 | (unsigned __int64)(v35 + 3);
        }
        sub_164B780((__int64)v35, v66);
        sub_12A86E0((__int64 *)&v74, (__int64)v35);
      }
      else
      {
        v35 = (__int64 *)sub_15A2B60((__int64 *)a4, v34, 0, 0, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      }
      v36 = *(_BYTE *)(sub_1456040(*(_QWORD *)(a3 + 32)) + 8) == 11;
      v69 = 257;
      v37 = *(_QWORD *)(a3 + 32);
      if ( v36 )
      {
        v61 = sub_1456040(v37);
        v59 = sub_16431D0(*v35);
        v47 = sub_16431D0(v61);
        if ( v59 < v47 )
        {
          v35 = (__int64 *)sub_12AA3B0((__int64 *)&v74, 0x26u, (__int64)v35, v61, (__int64)v68);
        }
        else if ( v59 > v47 )
        {
          v35 = (__int64 *)sub_12AA3B0((__int64 *)&v74, 0x24u, (__int64)v35, v61, (__int64)v68);
        }
      }
      else
      {
        v38 = sub_1456040(v37);
        v35 = (__int64 *)sub_12AA3B0((__int64 *)&v74, 0x2Au, (__int64)v35, v38, (__int64)v68);
      }
      v68[0] = (__int64)"cast.cmo";
      v69 = 259;
      sub_164B780((__int64)v35, v68);
      v39 = sub_1B19340(a3, (__int64)&v74, (__int64)v35, *(_QWORD **)(*(_QWORD *)(a1 + 16) + 112LL), v63, a7, a8, a9);
      v69 = 259;
      v68[0] = (__int64)"ind.escape";
      sub_164B780((__int64)v39, v68);
      v40 = v73;
      v66[0] = (__int64)v30;
      if ( !v73 )
        break;
      v15 = v73 - 1;
      v41 = v15 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v42 = &v71[2 * v41];
      v14 = *v42;
      if ( v30 == (_QWORD *)*v42 )
        goto LABEL_28;
      v16 = 1;
      v58 = 0;
      while ( v14 != -8 )
      {
        if ( v14 == -16 && !v58 )
          v58 = v42;
        v41 = v15 & (v16 + v41);
        v42 = &v71[2 * v41];
        v14 = *v42;
        if ( v30 == (_QWORD *)*v42 )
          goto LABEL_28;
        v16 = (unsigned int)(v16 + 1);
      }
      if ( v58 )
        v42 = v58;
      ++v70;
      v14 = (unsigned int)(v72 + 1);
      if ( 4 * (int)v14 >= 3 * v73 )
        goto LABEL_83;
      if ( v73 - HIDWORD(v72) - (unsigned int)v14 <= v73 >> 3 )
        goto LABEL_84;
LABEL_79:
      LODWORD(v72) = v14;
      if ( *v42 != -8 )
        --HIDWORD(v72);
      *v42 = (__int64)v30;
      v42[1] = 0;
LABEL_28:
      v42[1] = (__int64)v39;
      if ( !v74 )
        goto LABEL_19;
      sub_161E7C0((__int64)&v74, (__int64)v74);
      v29 = *(_QWORD *)(v29 + 8);
      if ( !v29 )
        goto LABEL_30;
    }
    ++v70;
LABEL_83:
    v40 = 2 * v73;
LABEL_84:
    sub_176F940((__int64)&v70, v40);
    sub_176A9A0((__int64)&v70, v66, v68);
    v42 = (__int64 *)v68[0];
    v30 = (_QWORD *)v66[0];
    v14 = (unsigned int)(v72 + 1);
    goto LABEL_79;
  }
LABEL_30:
  v43 = v71;
  if ( (_DWORD)v72 )
  {
    v48 = &v71[2 * v73];
    if ( v71 != v48 )
    {
      v49 = v71;
      while ( 1 )
      {
        v50 = *v49;
        v51 = v49;
        if ( *v49 != -8 && v50 != -16 )
          break;
        v49 += 2;
        if ( v48 == v49 )
          return j___libc_free_0(v43);
      }
      if ( v49 != v48 )
      {
        v52 = *v49;
        v53 = *(_DWORD *)(v50 + 20) & 0xFFFFFFF;
        if ( !v53 )
          goto LABEL_59;
        while ( 1 )
        {
          v54 = 0;
          v15 = *(_BYTE *)(v52 + 23) & 0x40;
          v16 = v52 - 24LL * v53;
          v55 = 24LL * *(unsigned int *)(v52 + 56) + 8;
          while ( 1 )
          {
            v14 = v52 - 24LL * v53;
            if ( (_BYTE)v15 )
              v14 = *(_QWORD *)(v52 - 8);
            if ( a6 == *(_QWORD *)(v14 + v55) )
              break;
            ++v54;
            v55 += 8;
            if ( v53 == v54 )
              goto LABEL_59;
          }
          while ( 1 )
          {
            v51 += 2;
            if ( v51 == v48 )
              goto LABEL_56;
            while ( 1 )
            {
              v52 = *v51;
              if ( *v51 != -16 && v52 != -8 )
                break;
              v51 += 2;
              if ( v48 == v51 )
                goto LABEL_56;
            }
            if ( v51 == v48 )
            {
LABEL_56:
              v43 = v71;
              return j___libc_free_0(v43);
            }
            v53 = *(_DWORD *)(v52 + 20) & 0xFFFFFFF;
            if ( v53 )
              break;
LABEL_59:
            sub_1704F80(v52, v51[1], a6, v14, v15, v16);
          }
        }
      }
    }
  }
  return j___libc_free_0(v43);
}
