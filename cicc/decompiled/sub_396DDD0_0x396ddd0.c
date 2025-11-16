// Function: sub_396DDD0
// Address: 0x396ddd0
//
void __fastcall sub_396DDD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  unsigned __int64 *v5; // r12
  __int64 *v7; // rsi
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r13
  int v10; // eax
  unsigned __int64 *v11; // r13
  unsigned __int64 *v12; // r12
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 *v15; // r12
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rsi
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // r13
  unsigned int v21; // r15d
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdi
  _BYTE *v26; // rax
  __int64 v27; // rdi
  _BYTE *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rbx
  unsigned __int64 v32; // r15
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int64 v38; // r9
  __int64 v39; // rax
  unsigned int v40; // eax
  __int64 v41; // rsi
  __int64 v42; // rdx
  unsigned __int64 v43; // r14
  __int64 v44; // rax
  __m128i *v45; // rdx
  __m128i si128; // xmm0
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned int v50; // esi
  int v51; // eax
  unsigned __int64 v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // rax
  int v55; // eax
  unsigned __int64 v56; // rax
  _QWORD *v57; // rax
  int v58; // eax
  __int64 v59; // [rsp+0h] [rbp-80h]
  __int64 v60; // [rsp+8h] [rbp-78h]
  unsigned __int64 v61; // [rsp+8h] [rbp-78h]
  __int64 v62; // [rsp+8h] [rbp-78h]
  __int64 v63; // [rsp+8h] [rbp-78h]
  __int64 v64; // [rsp+10h] [rbp-70h]
  __int64 v65; // [rsp+10h] [rbp-70h]
  unsigned __int64 v66; // [rsp+10h] [rbp-70h]
  __int64 v67; // [rsp+10h] [rbp-70h]
  unsigned __int64 v68; // [rsp+10h] [rbp-70h]
  __int64 v69; // [rsp+10h] [rbp-70h]
  __int64 v70; // [rsp+10h] [rbp-70h]
  __int64 v71; // [rsp+10h] [rbp-70h]
  __int64 v72; // [rsp+10h] [rbp-70h]
  void *v73; // [rsp+18h] [rbp-68h]
  int v74; // [rsp+18h] [rbp-68h]
  __int64 v75; // [rsp+18h] [rbp-68h]
  __int64 v76; // [rsp+18h] [rbp-68h]
  unsigned __int64 *v77; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v78; // [rsp+28h] [rbp-58h]
  char *v79; // [rsp+30h] [rbp-50h] BYREF
  __int64 v80; // [rsp+38h] [rbp-48h]
  _BYTE v81[64]; // [rsp+40h] [rbp-40h] BYREF

  v4 = a1 + 8;
  v5 = (unsigned __int64 *)&v77;
  v7 = (__int64 *)(a1 + 8);
  v73 = sub_16982C0();
  if ( *(void **)(a1 + 8) == v73 )
    sub_169D930((__int64)&v77, (__int64)v7);
  else
    sub_169D7E0((__int64)&v77, v7);
  if ( *(_BYTE *)(a3 + 416) )
  {
    v79 = v81;
    v80 = 0x800000000LL;
    if ( v73 == *(void **)(a1 + 8) )
      sub_16A4A90(v4, (__int64)&v79, 0, 3u, 1u);
    else
      sub_16A3760(v4, (__int64)&v79, 0, 3u, 1);
    v23 = **(_QWORD **)(a3 + 256);
    if ( a2 )
    {
      v24 = (*(__int64 (**)(void))(v23 + 112))();
      sub_154E060(a2, v24, 0, 0);
    }
    else
    {
      v44 = (*(__int64 (**)(void))(v23 + 112))();
      v45 = *(__m128i **)(v44 + 24);
      if ( *(_QWORD *)(v44 + 16) - (_QWORD)v45 <= 0x13u )
      {
        sub_16E7EE0(v44, "Printing <null> Type", 0x14u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4289F10);
        v45[1].m128i_i32[0] = 1701869908;
        *v45 = si128;
        *(_QWORD *)(v44 + 24) += 20LL;
      }
    }
    v25 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a3 + 256) + 112LL))(*(_QWORD *)(a3 + 256));
    v26 = *(_BYTE **)(v25 + 24);
    if ( (unsigned __int64)v26 >= *(_QWORD *)(v25 + 16) )
    {
      v25 = sub_16E7DE0(v25, 32);
    }
    else
    {
      *(_QWORD *)(v25 + 24) = v26 + 1;
      *v26 = 32;
    }
    v27 = sub_16E7EE0(v25, v79, (unsigned int)v80);
    v28 = *(_BYTE **)(v27 + 24);
    if ( (unsigned __int64)v28 >= *(_QWORD *)(v27 + 16) )
    {
      sub_16E7DE0(v27, 10);
    }
    else
    {
      *(_QWORD *)(v27 + 24) = v28 + 1;
      *v28 = 10;
    }
    if ( v79 != v81 )
      _libc_free((unsigned __int64)v79);
  }
  v8 = v78 >> 3;
  if ( v78 > 0x40 )
    v5 = v77;
  v74 = (v78 >> 3) & 7;
  if ( !*(_BYTE *)sub_396DDB0(a3) || *(_BYTE *)(a2 + 8) == 6 )
  {
    v20 = v8 >> 3;
    if ( v20 )
    {
      v21 = 0;
      v22 = 0;
      do
      {
        (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64))(**(_QWORD **)(a3 + 256) + 424LL))(
          *(_QWORD *)(a3 + 256),
          v5[v22],
          8);
        v22 = ++v21;
      }
      while ( v21 < v20 );
    }
    else
    {
      v22 = 0;
    }
    if ( v74 )
      (*(void (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a3 + 256) + 424LL))(
        *(_QWORD *)(a3 + 256),
        v5[v22]);
  }
  else
  {
    v9 = ((unsigned __int64)v78 + 63) >> 6;
    v10 = v9 - 1;
    if ( v74 )
    {
      (*(void (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a3 + 256) + 424LL))(
        *(_QWORD *)(a3 + 256),
        v5[v10]);
      v10 = v9 - 2;
    }
    if ( v10 >= 0 )
    {
      v11 = &v5[v10];
      v12 = v5 - 1;
      do
      {
        v13 = *v11--;
        (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a3 + 256) + 424LL))(
          *(_QWORD *)(a3 + 256),
          v13,
          8);
      }
      while ( v12 != v11 );
    }
  }
  v14 = sub_396DDB0(a3);
  v15 = *(__int64 **)(a3 + 256);
  v16 = 1;
  v17 = v14;
  v18 = a2;
  v19 = (unsigned int)sub_15A9FE0(v14, a2);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v18 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v34 = *(_QWORD *)(v18 + 32);
        v18 = *(_QWORD *)(v18 + 24);
        v16 *= v34;
        continue;
      case 1:
        v29 = 16;
        break;
      case 2:
        v29 = 32;
        break;
      case 3:
      case 9:
        v29 = 64;
        break;
      case 4:
        v29 = 80;
        break;
      case 5:
      case 6:
        v29 = 128;
        break;
      case 7:
        v29 = 8 * (unsigned int)sub_15A9520(v17, 0);
        break;
      case 0xB:
        v29 = *(_DWORD *)(v18 + 8) >> 8;
        break;
      case 0xD:
        v29 = 8LL * *(_QWORD *)sub_15A9930(v17, v18);
        break;
      case 0xE:
        v64 = *(_QWORD *)(v18 + 24);
        v75 = *(_QWORD *)(v18 + 32);
        v35 = sub_15A9FE0(v17, v64);
        v36 = v64;
        v37 = 1;
        v38 = v35;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v36 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v49 = *(_QWORD *)(v36 + 32);
              v36 = *(_QWORD *)(v36 + 24);
              v37 *= v49;
              continue;
            case 1:
              v47 = 16;
              goto LABEL_68;
            case 2:
              v47 = 32;
              goto LABEL_68;
            case 3:
            case 9:
              v47 = 64;
              goto LABEL_68;
            case 4:
              v47 = 80;
              goto LABEL_68;
            case 5:
            case 6:
              v47 = 128;
              goto LABEL_68;
            case 7:
              v60 = v37;
              v50 = 0;
              v66 = v38;
              goto LABEL_78;
            case 0xB:
              v47 = *(_DWORD *)(v36 + 8) >> 8;
              goto LABEL_68;
            case 0xD:
              v62 = v37;
              v68 = v38;
              v53 = (_QWORD *)sub_15A9930(v17, v36);
              v38 = v68;
              v37 = v62;
              v47 = 8LL * *v53;
              goto LABEL_68;
            case 0xE:
              v59 = v37;
              v61 = v38;
              v67 = *(_QWORD *)(v36 + 32);
              v52 = sub_12BE0A0(v17, *(_QWORD *)(v36 + 24));
              v38 = v61;
              v37 = v59;
              v47 = 8 * v67 * v52;
              goto LABEL_68;
            case 0xF:
              v60 = v37;
              v66 = v38;
              v50 = *(_DWORD *)(v36 + 8) >> 8;
LABEL_78:
              v51 = sub_15A9520(v17, v50);
              v38 = v66;
              v37 = v60;
              v47 = (unsigned int)(8 * v51);
LABEL_68:
              v29 = 8 * v38 * v75 * ((v38 + ((unsigned __int64)(v47 * v37 + 7) >> 3) - 1) / v38);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v29 = 8 * (unsigned int)sub_15A9520(v17, *(_DWORD *)(v18 + 8) >> 8);
        break;
    }
    break;
  }
  v30 = v16 * v29;
  v31 = 1;
  v32 = (v19 + ((unsigned __int64)(v30 + 7) >> 3) - 1) / v19 * v19;
  while ( 1 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 1:
        v33 = 16;
        goto LABEL_36;
      case 2:
        v33 = 32;
        goto LABEL_36;
      case 3:
      case 9:
        v33 = 64;
        goto LABEL_36;
      case 4:
        v33 = 80;
        goto LABEL_36;
      case 5:
      case 6:
        v33 = 128;
        goto LABEL_36;
      case 7:
        v33 = 8 * (unsigned int)sub_15A9520(v17, 0);
        goto LABEL_36;
      case 0xB:
        v33 = *(_DWORD *)(a2 + 8) >> 8;
        goto LABEL_36;
      case 0xD:
        v33 = 8LL * *(_QWORD *)sub_15A9930(v17, a2);
        goto LABEL_36;
      case 0xE:
        v65 = *(_QWORD *)(a2 + 24);
        v76 = *(_QWORD *)(a2 + 32);
        v40 = sub_15A9FE0(v17, v65);
        v41 = v65;
        v42 = 1;
        v43 = v40;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v41 + 8) )
          {
            case 1:
              v48 = 16;
              goto LABEL_71;
            case 2:
              v48 = 32;
              goto LABEL_71;
            case 3:
            case 9:
              v48 = 64;
              goto LABEL_71;
            case 4:
              v48 = 80;
              goto LABEL_71;
            case 5:
            case 6:
              v48 = 128;
              goto LABEL_71;
            case 7:
              v72 = v42;
              v58 = sub_15A9520(v17, 0);
              v42 = v72;
              v48 = (unsigned int)(8 * v58);
              goto LABEL_71;
            case 0xB:
              v48 = *(_DWORD *)(v41 + 8) >> 8;
              goto LABEL_71;
            case 0xD:
              v71 = v42;
              v57 = (_QWORD *)sub_15A9930(v17, v41);
              v42 = v71;
              v48 = 8LL * *v57;
              goto LABEL_71;
            case 0xE:
              v63 = v42;
              v70 = *(_QWORD *)(v41 + 32);
              v56 = sub_12BE0A0(v17, *(_QWORD *)(v41 + 24));
              v42 = v63;
              v48 = 8 * v70 * v56;
              goto LABEL_71;
            case 0xF:
              v69 = v42;
              v55 = sub_15A9520(v17, *(_DWORD *)(v41 + 8) >> 8);
              v42 = v69;
              v48 = (unsigned int)(8 * v55);
LABEL_71:
              v33 = 8 * v43 * v76 * ((v43 + ((unsigned __int64)(v48 * v42 + 7) >> 3) - 1) / v43);
              goto LABEL_36;
            case 0x10:
              v54 = *(_QWORD *)(v41 + 32);
              v41 = *(_QWORD *)(v41 + 24);
              v42 *= v54;
              continue;
            default:
              goto LABEL_93;
          }
        }
      case 0xF:
        v33 = 8 * (unsigned int)sub_15A9520(v17, *(_DWORD *)(a2 + 8) >> 8);
LABEL_36:
        sub_38DD110(v15, v32 - ((unsigned __int64)(v33 * v31 + 7) >> 3));
        if ( v78 > 0x40 )
        {
          if ( v77 )
            j_j___libc_free_0_0((unsigned __int64)v77);
        }
        return;
      case 0x10:
        v39 = *(_QWORD *)(a2 + 32);
        a2 = *(_QWORD *)(a2 + 24);
        v31 *= v39;
        break;
      default:
LABEL_93:
        BUG();
    }
  }
}
