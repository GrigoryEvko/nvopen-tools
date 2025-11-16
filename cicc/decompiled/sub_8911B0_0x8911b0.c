// Function: sub_8911B0
// Address: 0x8911b0
//
void __fastcall sub_8911B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // al
  char v7; // al
  char v8; // al
  char v9; // al
  char v10; // al
  char v11; // al
  char v12; // al
  __int64 v13; // r12
  char v16; // al
  char v17; // al
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int8 v23; // al
  bool v24; // zf
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  char v35; // cl
  char v36; // cl
  char v37; // al
  __int64 v38; // rax
  __int64 v39; // rsi
  char v40; // al
  char v41; // al
  char v42; // al
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rax

  v13 = *(_QWORD *)(a1 + 336);
  if ( v13 && a2 )
  {
    v16 = 2;
    if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
      v16 = ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 8) & 2) == 0) + 1;
    *(_BYTE *)(v13 + 88) = (16 * v16) | *(_BYTE *)(v13 + 88) & 0x8F;
    v17 = *(_BYTE *)(a2 + 80);
    switch ( v17 )
    {
      case 4:
      case 5:
        v19 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 80LL);
        goto LABEL_8;
      case 6:
LABEL_28:
        v24 = *(_QWORD *)(v13 + 200) == 0;
        *(_BYTE *)(v13 + 120) = 7;
        if ( v24 )
          *(_QWORD *)(v13 + 200) = v13;
        if ( *(_DWORD *)(a1 + 36) )
          *(_QWORD *)(*(_QWORD *)(v13 + 200) + 208LL) = v13;
        if ( dword_4F07590 )
          *(_QWORD *)(v13 + 192) = *(_QWORD *)(a2 + 88);
        else
          *(_QWORD *)(v13 + 192) = 0;
        v18 = *(_BYTE *)(a2 + 81);
        goto LABEL_15;
      case 9:
LABEL_41:
        *(_BYTE *)(v13 + 120) = 5;
        v26 = 0;
        if ( dword_4F07590 )
          v26 = *(_QWORD *)(a2 + 88);
        *(_QWORD *)(v13 + 192) = v26;
        v20 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 88) + 216LL) + 16LL);
        *(_QWORD *)(v13 + 200) = v20;
        if ( (*(_BYTE *)(*(_QWORD *)a1 + 127LL) & 4) == 0 && (*(_BYTE *)(*(_QWORD *)(a2 + 88) + 176LL) & 1) != 0 )
          goto LABEL_38;
        goto LABEL_14;
      case 10:
LABEL_35:
        *(_BYTE *)(v13 + 120) = 4;
        v25 = 0;
        if ( dword_4F07590 )
          v25 = *(_QWORD *)(a2 + 88);
        *(_QWORD *)(v13 + 192) = v25;
        v20 = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 248LL);
        *(_QWORD *)(v13 + 200) = v20;
        if ( *(_DWORD *)(a1 + 36) )
        {
LABEL_14:
          *(_QWORD *)(v20 + 208) = v13;
          v18 = *(_BYTE *)(a2 + 81);
LABEL_15:
          v21 = *(_QWORD *)(a2 + 64);
          if ( (v18 & 0x10) == 0 )
            goto LABEL_39;
        }
        else
        {
LABEL_38:
          v21 = *(_QWORD *)(a2 + 64);
          if ( (*(_BYTE *)(a2 + 81) & 0x10) == 0 )
          {
LABEL_39:
            if ( !v21 )
              goto LABEL_22;
            sub_877E90(0, v13, v21);
LABEL_19:
            if ( (*(_BYTE *)(a2 + 81) & 0x10) != 0 && *(_QWORD *)(a2 + 64) == *(_QWORD *)(a1 + 240) )
              *(_BYTE *)(v13 + 88) = *(_BYTE *)(a1 + 164) & 3 | *(_BYTE *)(v13 + 88) & 0xFC;
LABEL_22:
            *(_QWORD *)(v13 + 72) = sub_729420(1, (const __m128i *)(a1 + 344));
            v23 = *(_BYTE *)(a2 + 80);
            if ( v23 == 9 )
            {
              *(__m128i *)(v13 + 148) = _mm_loadu_si128((const __m128i *)(a1 + 408));
              goto LABEL_27;
            }
            if ( v23 <= 9u )
            {
              if ( (unsigned __int8)(v23 - 4) > 1u )
                goto LABEL_27;
            }
            else if ( v23 != 10 && (unsigned __int8)(v23 - 19) > 1u )
            {
              goto LABEL_27;
            }
            *(__m128i *)(v13 + 148) = _mm_loadu_si128((const __m128i *)(a1 + 472));
LABEL_27:
            sub_7344C0(v13, *(_DWORD *)(a1 + 204));
            return;
          }
        }
        v22 = *(_QWORD *)(*(_QWORD *)(v21 + 168) + 152LL);
        if ( !v22 || (*(_BYTE *)(v22 + 29) & 0x20) != 0 )
        {
          *(_BYTE *)(v13 + 89) &= ~4u;
          *(_QWORD *)(v13 + 40) = qword_4F07288;
        }
        else
        {
          sub_877E20(0, v13, v21, a4, a5, a6);
        }
        goto LABEL_19;
      case 19:
      case 20:
      case 21:
      case 22:
        v19 = *(_QWORD *)(a2 + 88);
        goto LABEL_8;
      default:
        v19 = 0;
LABEL_8:
        switch ( v17 )
        {
          case 4:
          case 5:
            *(_BYTE *)(v13 + 120) = 6;
            a4 = dword_4F07590;
            if ( dword_4F07590 )
              *(_QWORD *)(v13 + 192) = *(_QWORD *)(a2 + 88);
            else
              *(_QWORD *)(v13 + 192) = 0;
            v20 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 88) + 168LL) + 160LL);
            *(_QWORD *)(v13 + 200) = v20;
            if ( *(_DWORD *)(a1 + 36) )
              goto LABEL_14;
            goto LABEL_38;
          case 6:
            goto LABEL_28;
          case 9:
            goto LABEL_41;
          case 10:
            goto LABEL_35;
          case 19:
            *(_BYTE *)(v13 + 120) = 1;
            v32 = *(_QWORD *)(a2 + 88);
            a4 = *(_QWORD *)(v32 + 88);
            if ( a4 )
            {
              if ( (*(_BYTE *)(v32 + 160) & 1) != 0 )
                a4 = a2;
            }
            else
            {
              a4 = a2;
            }
            switch ( *(_BYTE *)(a4 + 80) )
            {
              case 4:
              case 5:
                v33 = *(_QWORD *)(*(_QWORD *)(a4 + 96) + 80LL);
                break;
              case 6:
                v33 = *(_QWORD *)(*(_QWORD *)(a4 + 96) + 32LL);
                break;
              case 9:
              case 0xA:
                v33 = *(_QWORD *)(*(_QWORD *)(a4 + 96) + 56LL);
                break;
              case 0x13:
              case 0x14:
              case 0x15:
              case 0x16:
                v33 = *(_QWORD *)(a4 + 88);
                break;
              default:
                v33 = 0;
                break;
            }
            if ( v19 != v33 )
              *(_QWORD *)(v13 + 216) = *(_QWORD *)(v33 + 104);
            a6 = dword_4F07590;
            if ( dword_4F07590 )
              *(_QWORD *)(v13 + 192) = *(_QWORD *)(*(_QWORD *)(v19 + 176) + 88LL);
            else
              *(_QWORD *)(v13 + 192) = 0;
            if ( (*(_BYTE *)(v19 + 160) & 2) != 0 )
            {
              *(_QWORD *)(v13 + 200) = v13;
              v18 = *(_BYTE *)(a2 + 81);
              goto LABEL_15;
            }
            v20 = *(_QWORD *)(v19 + 104);
            *(_QWORD *)(v13 + 200) = v20;
            a5 = *(unsigned int *)(a1 + 36);
            if ( (_DWORD)a5 && !*(_QWORD *)(v20 + 208) )
              goto LABEL_14;
            goto LABEL_38;
          case 20:
            *(_BYTE *)(v13 + 120) = 2;
            v30 = *(_QWORD *)(a2 + 88);
            v31 = *(_QWORD *)(v30 + 88);
            if ( v31 )
            {
              if ( (*(_BYTE *)(v30 + 160) & 1) != 0 )
                v31 = a2;
            }
            else
            {
              v31 = a2;
            }
            a4 = *(_QWORD *)(v19 + 176);
            switch ( *(_BYTE *)(v31 + 80) )
            {
              case 4:
              case 5:
                v39 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 80LL);
                break;
              case 6:
                v39 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 32LL);
                break;
              case 9:
              case 0xA:
                v39 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 56LL);
                break;
              case 0x13:
              case 0x14:
              case 0x15:
              case 0x16:
                v39 = *(_QWORD *)(v31 + 88);
                break;
              default:
                v6 = *(_BYTE *)(a4 + 198) & 8 | *(_BYTE *)(v13 + 184) & 0xF7;
                *(_BYTE *)(v13 + 184) = v6;
                v7 = *(_BYTE *)(a4 + 198) & 0x10 | v6 & 0xEF;
                *(_BYTE *)(v13 + 184) = v7;
                v8 = *(_BYTE *)(a4 + 198) & 0x20 | v7 & 0xDF;
                *(_BYTE *)(v13 + 184) = v8;
                v9 = *(_BYTE *)(a4 + 198) & 0x40 | v8 & 0xBF;
                *(_BYTE *)(v13 + 184) = v9;
                *(_BYTE *)(v13 + 184) = *(_BYTE *)(a4 + 198) & 0x80 | v9 & 0x7F;
                BUG();
            }
            v40 = *(_BYTE *)(a4 + 198) & 8 | *(_BYTE *)(v13 + 184) & 0xF7;
            *(_BYTE *)(v13 + 184) = v40;
            v41 = *(_BYTE *)(a4 + 198) & 0x10 | v40 & 0xEF;
            *(_BYTE *)(v13 + 184) = v41;
            v42 = *(_BYTE *)(a4 + 198) & 0x20 | v41 & 0xDF;
            *(_BYTE *)(v13 + 184) = v42;
            v43 = *(_BYTE *)(a4 + 198) & 0x40 | v42 & 0xBF;
            *(_BYTE *)(v13 + 184) = v43;
            *(_BYTE *)(v13 + 184) = *(_BYTE *)(a4 + 198) & 0x80 | v43 & 0x7F;
            if ( v19 != v39 )
              *(_QWORD *)(v13 + 216) = *(_QWORD *)(v39 + 104);
            v44 = 0;
            if ( dword_4F07590 )
              v44 = a4;
            *(_QWORD *)(v13 + 192) = v44;
            v45 = *(_QWORD *)(v19 + 104);
            *(_QWORD *)(v13 + 200) = v45;
            if ( *(_DWORD *)(a1 + 36) )
              *(_QWORD *)(v45 + 208) = v13;
            if ( *(_BYTE *)(a4 + 172) != 2 )
              goto LABEL_38;
            *(_BYTE *)(v13 + 88) = *(_BYTE *)(v13 + 88) & 0x8F | 0x10;
            v18 = *(_BYTE *)(a2 + 81);
            goto LABEL_15;
          case 21:
            *(_BYTE *)(v13 + 120) = 3;
            v27 = *(_QWORD *)(a2 + 88);
            v28 = *(_QWORD *)(v27 + 88);
            if ( v28 )
            {
              if ( (*(_BYTE *)(v27 + 160) & 1) != 0 )
                v28 = a2;
            }
            else
            {
              v28 = a2;
            }
            v29 = *(_QWORD *)(v19 + 192);
            switch ( *(_BYTE *)(v28 + 80) )
            {
              case 4:
              case 5:
                v34 = *(_QWORD *)(*(_QWORD *)(v28 + 96) + 80LL);
                break;
              case 6:
                v34 = *(_QWORD *)(*(_QWORD *)(v28 + 96) + 32LL);
                break;
              case 9:
              case 0xA:
                v34 = *(_QWORD *)(*(_QWORD *)(v28 + 96) + 56LL);
                break;
              case 0x13:
              case 0x14:
              case 0x15:
              case 0x16:
                v34 = *(_QWORD *)(v28 + 88);
                break;
              default:
                v10 = *(_BYTE *)(v29 + 157) & 1 | *(_BYTE *)(v13 + 184) & 0xFE;
                *(_BYTE *)(v13 + 184) = v10;
                v11 = *(_BYTE *)(v29 + 156) & 4 | v10 & 0xFB;
                *(_BYTE *)(v13 + 184) = v11;
                v12 = (16 * (*(_BYTE *)(v29 + 156) & 1)) | v11 & 0xEF;
                *(_BYTE *)(v13 + 184) = v12;
                *(_BYTE *)(v13 + 184) = *(_BYTE *)(v29 + 156) & 2 | v12 & 0xFD;
                BUG();
            }
            v35 = *(_BYTE *)(v29 + 157) & 1 | *(_BYTE *)(v13 + 184) & 0xFE;
            *(_BYTE *)(v13 + 184) = v35;
            v36 = *(_BYTE *)(v29 + 156) & 4 | v35 & 0xFB;
            *(_BYTE *)(v13 + 184) = v36;
            v37 = v36 & 0xEF | (16 * (*(_BYTE *)(v29 + 156) & 1));
            *(_BYTE *)(v13 + 184) = v37;
            a4 = *(_BYTE *)(v29 + 156) & 2;
            *(_BYTE *)(v13 + 184) = a4 | v37 & 0xFD;
            if ( v19 != v34 )
              *(_QWORD *)(v13 + 216) = *(_QWORD *)(v34 + 104);
            a5 = dword_4F07590;
            v38 = 0;
            if ( dword_4F07590 )
              v38 = v29;
            *(_QWORD *)(v13 + 192) = v38;
            v20 = *(_QWORD *)(*(_QWORD *)(v29 + 216) + 16LL);
            *(_QWORD *)(v13 + 200) = v20;
            a6 = *(unsigned int *)(a1 + 36);
            if ( (!(_DWORD)a6 || *(_QWORD *)(a1 + 240)) && (*(_BYTE *)(v29 + 176) & 1) == 0 )
              goto LABEL_38;
            goto LABEL_14;
          case 22:
            *(_BYTE *)(v13 + 120) = 9;
            *(_QWORD *)(v13 + 200) = v13;
            *(_QWORD *)(v13 + 208) = v13;
            goto LABEL_38;
          default:
            v18 = *(_BYTE *)(a2 + 81);
            if ( (v18 & 0x20) != 0 )
              goto LABEL_15;
            return;
        }
    }
  }
}
