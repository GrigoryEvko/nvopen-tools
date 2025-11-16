// Function: sub_21CAE40
// Address: 0x21cae40
//
void __fastcall sub_21CAE40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v10; // r8d
  int v11; // r9d
  char v12; // al
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 v15; // r12
  __int64 v16; // r14
  _QWORD *v17; // rbx
  __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // r8
  __int64 v21; // r11
  __int64 v22; // rdx
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // r15
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // rax
  __int64 v35; // r14
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 v40; // rdx
  int v41; // r15d
  unsigned int v42; // r12d
  int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  int v49; // r12d
  __int64 v50; // r15
  __int64 v51; // rbx
  __int64 v52; // r14
  __int64 v53; // r13
  __int64 v54; // rax
  unsigned int v55; // esi
  int v56; // eax
  unsigned __int64 v57; // rax
  _QWORD *v58; // rax
  const void *v59; // [rsp+8h] [rbp-228h]
  __int64 v60; // [rsp+8h] [rbp-228h]
  const void *v61; // [rsp+10h] [rbp-220h]
  unsigned __int64 v62; // [rsp+10h] [rbp-220h]
  __int64 v63; // [rsp+10h] [rbp-220h]
  __int64 v64; // [rsp+10h] [rbp-220h]
  __int64 v65; // [rsp+10h] [rbp-220h]
  __int64 v66; // [rsp+18h] [rbp-218h]
  __int64 v67; // [rsp+18h] [rbp-218h]
  __int64 v68; // [rsp+18h] [rbp-218h]
  int v69; // [rsp+18h] [rbp-218h]
  __int64 v70; // [rsp+18h] [rbp-218h]
  __int64 *v72; // [rsp+30h] [rbp-200h]
  __int64 v73; // [rsp+30h] [rbp-200h]
  unsigned int v74; // [rsp+30h] [rbp-200h]
  __int64 v75; // [rsp+30h] [rbp-200h]
  __int64 v76; // [rsp+38h] [rbp-1F8h]
  __m128i v77; // [rsp+40h] [rbp-1F0h] BYREF
  __m128i v78; // [rsp+50h] [rbp-1E0h] BYREF
  unsigned __int64 v79[2]; // [rsp+60h] [rbp-1D0h] BYREF
  _BYTE v80[128]; // [rsp+70h] [rbp-1C0h] BYREF
  _BYTE *v81; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v82; // [rsp+F8h] [rbp-138h]
  _BYTE v83[304]; // [rsp+100h] [rbp-130h] BYREF

  v81 = v83;
  v79[0] = (unsigned __int64)v80;
  v82 = 0x1000000000LL;
  v79[1] = 0x1000000000LL;
  if ( !sub_1642F90(a3, 128) )
  {
    v12 = *(_BYTE *)(a3 + 8);
    if ( v12 == 13 )
    {
      v13 = sub_15A9930(a2, a3);
      v14 = *(__int64 **)(a3 + 16);
      v72 = &v14[*(unsigned int *)(a3 + 12)];
      if ( v72 != v14 )
      {
        v66 = a6;
        v15 = a5;
        v16 = a4;
        v17 = (_QWORD *)(v13 + 16);
        do
        {
          v18 = *v14++;
          sub_21CAE40(a1, a2, v18, v16, v15, *v17++ + v66);
        }
        while ( v72 != v14 );
      }
      goto LABEL_6;
    }
    if ( v12 == 14 )
    {
      v67 = a3;
      v73 = *(_QWORD *)(a3 + 24);
      v19 = sub_15A9FE0(a2, v73);
      v20 = v73;
      v21 = v67;
      v22 = 1;
      v23 = v19;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v20 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v54 = *(_QWORD *)(v20 + 32);
            v20 = *(_QWORD *)(v20 + 24);
            v22 *= v54;
            continue;
          case 1:
            v47 = 16;
            goto LABEL_54;
          case 2:
            v47 = 32;
            goto LABEL_54;
          case 3:
          case 9:
            v47 = 64;
            goto LABEL_54;
          case 4:
            v47 = 80;
            goto LABEL_54;
          case 5:
          case 6:
            v47 = 128;
            goto LABEL_54;
          case 7:
            v63 = v22;
            v55 = 0;
            goto LABEL_68;
          case 0xB:
            v47 = *(_DWORD *)(v20 + 8) >> 8;
            goto LABEL_54;
          case 0xD:
            v65 = v22;
            v58 = (_QWORD *)sub_15A9930(a2, v20);
            v21 = v67;
            v22 = v65;
            v47 = 8LL * *v58;
            goto LABEL_54;
          case 0xE:
            v60 = v22;
            v64 = v67;
            v70 = *(_QWORD *)(v20 + 32);
            v57 = sub_12BE0A0(a2, *(_QWORD *)(v20 + 24));
            v21 = v64;
            v22 = v60;
            v47 = 8 * v70 * v57;
            goto LABEL_54;
          case 0xF:
            v63 = v22;
            v55 = *(_DWORD *)(v20 + 8) >> 8;
LABEL_68:
            v56 = sub_15A9520(a2, v55);
            v21 = v67;
            v22 = v63;
            v47 = (unsigned int)(8 * v56);
LABEL_54:
            v62 = v23 * ((v23 + ((unsigned __int64)(v47 * v22 + 7) >> 3) - 1) / v23);
            v69 = *(_QWORD *)(v21 + 32);
            if ( v69 )
            {
              v48 = a6;
              v49 = 0;
              v50 = a4;
              v51 = a5;
              v52 = a2;
              v53 = v48;
              do
              {
                ++v49;
                sub_21CAE40(a1, v52, v73, v50, v51, v53);
                v53 += v62;
              }
              while ( v69 != v49 );
            }
            break;
        }
        goto LABEL_6;
      }
    }
    v31 = 0;
    sub_20C7CE0(a1, a2, a3, (__int64)&v81, (__int64)v79, a6);
    v76 = 8LL * (unsigned int)v82;
    v59 = (const void *)(a5 + 16);
    v61 = (const void *)(a4 + 16);
    if ( !(_DWORD)v82 )
      goto LABEL_6;
    v34 = a5;
    v35 = a4;
    v36 = v34;
    while ( 1 )
    {
      v77 = _mm_loadu_si128((const __m128i *)&v81[2 * v31]);
      v39 = *(_QWORD *)(v79[0] + v31);
      if ( v77.m128i_i8[0] )
      {
        if ( (unsigned __int8)(v77.m128i_i8[0] - 14) <= 0x5Fu )
        {
          v74 = (unsigned __int16)word_435D740[(unsigned __int8)(v77.m128i_i8[0] - 14)];
          switch ( v77.m128i_i8[0] )
          {
            case 0x18:
            case 0x19:
            case 0x1A:
            case 0x1B:
            case 0x1C:
            case 0x1D:
            case 0x1E:
            case 0x1F:
            case 0x20:
            case 0x3E:
            case 0x3F:
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
              v78.m128i_i8[0] = 3;
              v78.m128i_i64[1] = 0;
              goto LABEL_39;
            case 0x21:
            case 0x22:
            case 0x23:
            case 0x24:
            case 0x25:
            case 0x26:
            case 0x27:
            case 0x28:
            case 0x44:
            case 0x45:
            case 0x46:
            case 0x47:
            case 0x48:
            case 0x49:
              v78.m128i_i8[0] = 4;
              v78.m128i_i64[1] = 0;
              goto LABEL_39;
            case 0x29:
            case 0x2A:
            case 0x2B:
            case 0x2C:
            case 0x2D:
            case 0x2E:
            case 0x2F:
            case 0x30:
            case 0x4A:
            case 0x4B:
            case 0x4C:
            case 0x4D:
            case 0x4E:
            case 0x4F:
              v78.m128i_i8[0] = 5;
              v78.m128i_i64[1] = 0;
              goto LABEL_39;
            case 0x31:
            case 0x32:
            case 0x33:
            case 0x34:
            case 0x35:
            case 0x36:
            case 0x50:
            case 0x51:
            case 0x52:
            case 0x53:
            case 0x54:
            case 0x55:
              v78.m128i_i8[0] = 6;
              v78.m128i_i64[1] = 0;
              goto LABEL_39;
            case 0x37:
              v78.m128i_i8[0] = 7;
              v78.m128i_i64[1] = 0;
              goto LABEL_39;
            case 0x56:
            case 0x57:
            case 0x58:
            case 0x62:
            case 0x63:
            case 0x64:
              v78.m128i_i8[0] = 8;
              v78.m128i_i64[1] = 0;
              goto LABEL_37;
            case 0x59:
            case 0x5A:
            case 0x5B:
            case 0x5C:
            case 0x5D:
            case 0x65:
            case 0x66:
            case 0x67:
            case 0x68:
            case 0x69:
              v78.m128i_i8[0] = 9;
              v78.m128i_i64[1] = 0;
              goto LABEL_39;
            case 0x5E:
            case 0x5F:
            case 0x60:
            case 0x61:
            case 0x6A:
            case 0x6B:
            case 0x6C:
            case 0x6D:
              v78.m128i_i8[0] = 10;
              v78.m128i_i64[1] = 0;
              goto LABEL_39;
            default:
              v78.m128i_i8[0] = 2;
              v78.m128i_i64[1] = 0;
              goto LABEL_39;
          }
        }
LABEL_27:
        v37 = *(unsigned int *)(v35 + 8);
        if ( (unsigned int)v37 >= *(_DWORD *)(v35 + 12) )
        {
          sub_16CD150(v35, v61, 0, 16, v32, v33);
          v37 = *(unsigned int *)(v35 + 8);
        }
        *(__m128i *)(*(_QWORD *)v35 + 16 * v37) = _mm_load_si128(&v77);
        ++*(_DWORD *)(v35 + 8);
        if ( v36 )
        {
          v38 = *(unsigned int *)(v36 + 8);
          if ( (unsigned int)v38 >= *(_DWORD *)(v36 + 12) )
          {
            sub_16CD150(v36, v59, 0, 8, v32, v33);
            v38 = *(unsigned int *)(v36 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v36 + 8 * v38) = v39;
          ++*(_DWORD *)(v36 + 8);
        }
        goto LABEL_33;
      }
      if ( !sub_1F58D20((__int64)&v77) )
        goto LABEL_27;
      v74 = sub_1F58D30((__int64)&v77);
      v78.m128i_i8[0] = sub_1F596B0((__int64)&v77);
      v78.m128i_i64[1] = v40;
      if ( v78.m128i_i8[0] == 8 )
      {
LABEL_37:
        if ( (v74 & 1) != 0 )
          goto LABEL_40;
        v78.m128i_i8[0] = 86;
        v78.m128i_i64[1] = 0;
        v74 >>= 1;
      }
LABEL_39:
      if ( v74 )
      {
LABEL_40:
        v68 = v31;
        v41 = 0;
        v42 = v74;
        do
        {
          v46 = *(unsigned int *)(v35 + 8);
          if ( (unsigned int)v46 >= *(_DWORD *)(v35 + 12) )
          {
            sub_16CD150(v35, v61, 0, 16, v32, v33);
            v46 = *(unsigned int *)(v35 + 8);
          }
          *(__m128i *)(*(_QWORD *)v35 + 16 * v46) = _mm_load_si128(&v78);
          ++*(_DWORD *)(v35 + 8);
          if ( v36 )
          {
            if ( v78.m128i_i8[0] )
              v43 = sub_1F3E310(&v78);
            else
              v43 = sub_1F58D40((__int64)&v78);
            v44 = *(unsigned int *)(v36 + 8);
            v45 = v39 + v41 * ((unsigned int)(v43 + 7) >> 3);
            if ( (unsigned int)v44 >= *(_DWORD *)(v36 + 12) )
            {
              v75 = v45;
              sub_16CD150(v36, v59, 0, 8, v32, v33);
              v44 = *(unsigned int *)(v36 + 8);
              v45 = v75;
            }
            *(_QWORD *)(*(_QWORD *)v36 + 8 * v44) = v45;
            ++*(_DWORD *)(v36 + 8);
          }
          ++v41;
        }
        while ( v41 != v42 );
        v31 = v68;
      }
LABEL_33:
      v31 += 8;
      if ( v76 == v31 )
        goto LABEL_6;
    }
  }
  v24 = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)v24 >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v10, v11);
    v24 = *(unsigned int *)(a4 + 8);
  }
  v25 = (_QWORD *)(*(_QWORD *)a4 + 16 * v24);
  *v25 = 6;
  v25[1] = 0;
  v26 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
  *(_DWORD *)(a4 + 8) = v26;
  if ( *(_DWORD *)(a4 + 12) <= (unsigned int)v26 )
  {
    sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v10, v11);
    v26 = *(unsigned int *)(a4 + 8);
  }
  v27 = (_QWORD *)(*(_QWORD *)a4 + 16 * v26);
  *v27 = 6;
  v27[1] = 0;
  ++*(_DWORD *)(a4 + 8);
  if ( a5 )
  {
    v28 = *(unsigned int *)(a5 + 8);
    if ( (unsigned int)v28 >= *(_DWORD *)(a5 + 12) )
    {
      sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v10, v11);
      v28 = *(unsigned int *)(a5 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a5 + 8 * v28) = a6;
    v29 = a6 + 8;
    v30 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
    *(_DWORD *)(a5 + 8) = v30;
    if ( *(_DWORD *)(a5 + 12) <= (unsigned int)v30 )
    {
      sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v10, v11);
      v30 = *(unsigned int *)(a5 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a5 + 8 * v30) = v29;
    ++*(_DWORD *)(a5 + 8);
  }
LABEL_6:
  if ( (_BYTE *)v79[0] != v80 )
    _libc_free(v79[0]);
  if ( v81 != v83 )
    _libc_free((unsigned __int64)v81);
}
