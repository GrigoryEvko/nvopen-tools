// Function: sub_8950B0
// Address: 0x8950b0
//
unsigned int *__fastcall sub_8950B0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  unsigned __int64 v4; // r12
  char v5; // al
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // r13
  unsigned int *result; // rax
  char v10; // si
  bool v11; // cl
  unsigned int *v12; // r10
  bool v13; // r13
  int v14; // eax
  unsigned int v15; // esi
  __int64 v16; // rax
  bool v17; // cl
  __int64 v18; // r10
  __int64 v19; // r13
  bool v20; // cl
  unsigned int *v21; // r10
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int8 v27; // cl
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 *v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  _BOOL4 v35; // eax
  __int64 v36; // rdi
  unsigned int *v37; // [rsp+8h] [rbp-48h]
  bool v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+10h] [rbp-40h]
  unsigned int *v40; // [rsp+10h] [rbp-40h]
  unsigned int *v41; // [rsp+10h] [rbp-40h]
  bool v42; // [rsp+10h] [rbp-40h]
  bool v43; // [rsp+18h] [rbp-38h]
  bool v44; // [rsp+18h] [rbp-38h]
  unsigned __int8 v45; // [rsp+18h] [rbp-38h]
  bool v46; // [rsp+18h] [rbp-38h]
  unsigned int *v47; // [rsp+18h] [rbp-38h]

  while ( 2 )
  {
    switch ( *(_BYTE *)(a1 + 80) )
    {
      case 4:
      case 5:
        v2 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
        goto LABEL_3;
      case 6:
        v2 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
        goto LABEL_3;
      case 9:
      case 0xA:
        v2 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
        goto LABEL_3;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v2 = *(_QWORD *)(a1 + 88);
        if ( !dword_4F077BC )
          goto LABEL_19;
        goto LABEL_4;
      default:
        v2 = 0;
LABEL_3:
        if ( !dword_4F077BC )
          goto LABEL_19;
LABEL_4:
        if ( qword_4F077A8 <= 0x76BFu )
          goto LABEL_5;
LABEL_19:
        if ( unk_4D0453C )
        {
LABEL_5:
          if ( sub_67D810((unsigned int *)(a1 + 48)) )
          {
            v3 = *(_QWORD *)(v2 + 104);
            if ( (*(_BYTE *)(v3 + 184) & 0x10) == 0 )
            {
              *(_BYTE *)(v3 + 136) = 1;
              *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v2 + 104) + 208LL) + 136LL) = 1;
            }
          }
        }
        v4 = *(_QWORD *)(v2 + 176);
        v5 = *(_BYTE *)(v2 + 424) | 2;
        v6 = *(_QWORD *)v4;
        *(_BYTE *)(v2 + 424) = v5;
        if ( (v5 & 4) == 0 )
          sub_894C00(v6);
        v7 = *(_QWORD *)(v6 + 96);
        *(_QWORD *)(v7 + 48) = sub_8807C0(v6);
        if ( (*(_BYTE *)(v2 + 160) & 1) == 0 )
        {
          v8 = *(_QWORD *)(v2 + 400);
          if ( !v8 )
          {
LABEL_22:
            v10 = 1;
            v11 = 1;
            if ( *(_BYTE *)(a1 + 80) == 10 )
            {
              result = (unsigned int *)(776LL * dword_4F04C64 + qword_4F04C68[0]);
              if ( (unsigned __int8)(*((_BYTE *)result + 4) - 6) <= 1u )
                goto LABEL_38;
            }
LABEL_23:
            result = *(unsigned int **)(v2 + 88);
            if ( !result || !v10 || *(_QWORD *)(v2 + 240) )
              goto LABEL_26;
            result = (unsigned int *)*((_QWORD *)result + 11);
            v12 = result + 46;
            goto LABEL_27;
          }
          switch ( *(_BYTE *)(v8 + 80) )
          {
            case 4:
            case 5:
              result = *(unsigned int **)(*(_QWORD *)(v8 + 96) + 80LL);
              goto LABEL_14;
            case 6:
              result = *(unsigned int **)(*(_QWORD *)(v8 + 96) + 32LL);
              goto LABEL_14;
            case 9:
            case 0xA:
              result = *(unsigned int **)(*(_QWORD *)(v8 + 96) + 56LL);
              goto LABEL_14;
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
              result = *(unsigned int **)(v8 + 88);
              if ( (*(_BYTE *)(v8 + 81) & 2) == 0 )
                goto LABEL_22;
              goto LABEL_15;
            default:
              result = 0;
LABEL_14:
              if ( (*(_BYTE *)(v8 + 81) & 2) == 0 )
                goto LABEL_22;
LABEL_15:
              if ( (result[106] & 2) != 0 )
                return result;
              result = (unsigned int *)sub_893570(*(_QWORD *)(v2 + 400));
              if ( !(_DWORD)result )
                return result;
              a1 = v8;
              break;
          }
          continue;
        }
        v11 = 1;
        result = (unsigned int *)qword_4F04C68[0];
        if ( *(_BYTE *)(a1 + 80) == 10 )
        {
          v10 = 0;
          result = (unsigned int *)(776LL * dword_4F04C64 + qword_4F04C68[0]);
          if ( (unsigned __int8)(*((_BYTE *)result + 4) - 6) <= 1u )
          {
LABEL_38:
            v11 = *((_QWORD *)result + 26) != *(_QWORD *)(a1 + 64);
            goto LABEL_23;
          }
        }
LABEL_26:
        v12 = (unsigned int *)(v2 + 184);
LABEL_27:
        if ( (*(_BYTE *)(v4 + 193) & 0x20) == 0 && !*(_DWORD *)(v4 + 160) && !*(_QWORD *)(v4 + 344) )
        {
          result = &dword_4F077BC;
          v13 = (*(_BYTE *)(*(_QWORD *)(v2 + 176) + 206LL) & 2) != 0;
          if ( !dword_4F077BC
            || (result = (unsigned int *)&qword_4F077A8, qword_4F077A8 > 0x76BFu)
            || (v41 = v12,
                v46 = v11,
                result = (unsigned int *)sub_67D810((unsigned int *)(a1 + 48)),
                v11 = v46,
                v12 = v41,
                !(_DWORD)result)
            || (result = *(unsigned int **)(v2 + 104), (result[46] & 0x10) != 0) )
          {
            if ( (v12[16] & 0x18) == 0 )
            {
              v14 = *(char *)(v2 + 160);
              v15 = ((v14 >> 31) & 0x1FFE) + 2;
              if ( *(_BYTE *)(v4 + 172) != 2 )
              {
                *(_BYTE *)(v4 + 172) = 0;
                *(_BYTE *)(v4 + 88) = *(_BYTE *)(v4 + 88) & 0x8F | 0x20;
                LOBYTE(v14) = *(_BYTE *)(v2 + 160);
              }
              v37 = v12;
              v38 = v11;
              if ( (v14 & 1) != 0 )
                v15 |= 0x400000u;
              if ( v13 )
                v15 |= 0x200000u;
              v16 = sub_892400(v2);
              v17 = v38;
              v18 = (__int64)v37;
              v19 = v16;
              if ( v38 )
              {
                v35 = sub_864700(*(_QWORD *)(v16 + 32), 0, v4, v6, a1, *(_QWORD *)(v4 + 240), 1, v15);
                v18 = (__int64)v37;
                v17 = v35;
              }
              v39 = v18;
              v43 = v17;
              sub_854C10(*(const __m128i **)(v2 + 56));
              sub_64F530(v6);
              v20 = v43;
              v21 = (unsigned int *)v39;
              if ( !dword_4D048B8 )
              {
                v36 = v39;
                v42 = v43;
                v47 = v21;
                sub_64A410(v36);
                v20 = v42;
                v21 = v47;
              }
              v44 = v20;
              v40 = v21;
              sub_7BC160(v19);
              sub_71E0E0(v4, (__int64)v40, 28, v22, v23);
              v27 = v44;
              if ( word_4F06418[0] == 74 )
              {
                sub_7B8B50(v4, v40, v24, v44, v25, v26);
                v27 = v44;
              }
              v45 = v27;
              sub_854980(v6, 0);
              if ( v45 )
                sub_863FE0(v6, 0, v28, v45, v29, v30);
              sub_8CBAA0(v4);
              while ( word_4F06418[0] != 9 )
                sub_7B8B50(v4, 0, v31, v32, v33, v34);
              return (unsigned int *)sub_7B8B50(v4, 0, v31, v32, v33, v34);
            }
          }
        }
        return result;
    }
  }
}
