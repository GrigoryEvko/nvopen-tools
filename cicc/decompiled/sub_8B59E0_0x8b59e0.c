// Function: sub_8B59E0
// Address: 0x8b59e0
//
__int64 __fastcall sub_8B59E0(__m128i *a1, __int64 a2, _QWORD *a3, unsigned int a4, int a5)
{
  _QWORD *v8; // r12
  unsigned int v9; // ebx
  __int64 v10; // r8
  unsigned int v11; // r14d
  _QWORD **v13; // rax
  __int64 *v14; // r11
  int v15; // ebx
  char v16; // dl
  __int64 v17; // rdi
  bool v18; // cl
  char v19; // r8
  __int64 *v20; // rdx
  unsigned int v21; // eax
  _QWORD *v22; // r8
  __int64 v23; // rcx
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rsi
  int v27; // eax
  unsigned int v28; // eax
  _BYTE *v29; // rcx
  __int64 v30; // rax
  _BYTE *v31; // [rsp+8h] [rbp-C8h]
  _BYTE v32[4]; // [rsp+20h] [rbp-B0h] BYREF
  int v33; // [rsp+24h] [rbp-ACh] BYREF
  _QWORD *v34; // [rsp+28h] [rbp-A8h] BYREF
  __int64 *v35; // [rsp+30h] [rbp-A0h] BYREF
  __int64 **v36; // [rsp+38h] [rbp-98h] BYREF
  __m128i v37[9]; // [rsp+40h] [rbp-90h] BYREF

  v8 = a3;
  v9 = a4;
  if ( dword_4F077BC && qword_4F077A8 < 0x9C40u )
    v9 = a4 & 0xFFFFFFF7;
  switch ( *(_BYTE *)(a2 + 80) )
  {
    case 4:
    case 5:
      v10 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 80LL);
      goto LABEL_6;
    case 6:
      v10 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 32LL);
      goto LABEL_6;
    case 9:
    case 0xA:
      v10 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 56LL);
      if ( !a3 )
        goto LABEL_11;
      goto LABEL_7;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v10 = *(_QWORD *)(a2 + 88);
      goto LABEL_6;
    default:
      v10 = 0;
LABEL_6:
      if ( a3 )
      {
LABEL_7:
        if ( a3 == 0 || a1 == 0 )
          return 0;
      }
      else
      {
LABEL_11:
        v13 = *(_QWORD ***)(v10 + 328);
        if ( !v13 )
          return 0;
        v8 = *v13;
        if ( *v13 == 0 || a1 == 0 )
          return 0;
      }
      v11 = sub_8AF210(a1, v8, v9, a2, v10, a5);
      if ( !v11 )
        return 0;
      sub_89A1A0(v8, a1->m128i_i64, &v34, &v35);
      v14 = v35;
      if ( v35 )
      {
        v15 = v9 & 8;
        while ( 1 )
        {
          v16 = *((_BYTE *)v14 + 8);
          if ( !v15 )
            goto LABEL_30;
LABEL_17:
          if ( !v16 )
          {
            if ( v14[4] )
              goto LABEL_19;
            goto LABEL_45;
          }
          if ( v16 == 1 )
            break;
          if ( v14[4] )
          {
LABEL_30:
            while ( v16 != 1 )
            {
              if ( v16 != 2 )
              {
LABEL_19:
                if ( !dword_4F077BC
                  || qword_4F077A8 <= 0x76BFu
                  || dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
                {
                  goto LABEL_26;
                }
                v17 = v14[4];
                v18 = 0;
                v19 = *(_BYTE *)(v17 + 140);
                if ( (unsigned __int8)(v19 - 9) <= 2u )
                  v18 = (*(_BYTE *)(*(_QWORD *)(v17 + 168) + 109LL) & 0x20) != 0;
                if ( *(_QWORD *)(v17 + 8) )
                {
LABEL_25:
                  if ( qword_4F077A8 <= 0x9CA3u )
                  {
LABEL_26:
                    v20 = v14;
                    goto LABEL_27;
                  }
LABEL_60:
                  if ( !v18 )
                  {
                    if ( (unsigned int)sub_8DD0E0(v17, v32, &v33, &v36, v37) && v33 )
                      v11 = 0;
                    v14 = v35;
                  }
                  goto LABEL_26;
                }
                if ( qword_4F077A8 > 0x9E33u && (*(_BYTE *)(v17 + 89) & 4) != 0 )
                {
                  v11 = 1;
                  goto LABEL_60;
                }
                if ( (v18 || (unsigned __int8)(v19 - 9) > 2u) && (v19 != 2 || (*(_BYTE *)(v17 + 161) & 8) == 0) )
                  goto LABEL_25;
LABEL_52:
                *((_BYTE *)v14 + 25) &= ~1u;
                sub_89A1C0((__int64 *)&v34, &v35);
                return 0;
              }
              v21 = sub_8B2F00((__int64)v14, v34[8], (__int64)a1, (__int64)v8, 0, (__int64 *)(a2 + 48));
              v20 = v35;
              v11 = v21;
LABEL_27:
              *((_BYTE *)v20 + 25) &= ~1u;
              sub_89A1C0((__int64 *)&v34, &v35);
              v14 = v35;
              if ( !v35 )
                return v11;
              if ( !v11 )
                return 0;
              v16 = *((_BYTE *)v35 + 8);
              if ( v15 )
                goto LABEL_17;
            }
LABEL_38:
            v22 = v34;
            v23 = *((_BYTE *)v34 + 57) & 8;
            if ( (*((_BYTE *)v34 + 57) & 8) != 0 || (v34[9] & 1) == 0 )
            {
              v24 = *(_QWORD *)(v34[8] + 128LL);
            }
            else
            {
              v33 = 0;
              sub_892150(v37);
              v36 = sub_8A2270(*(_QWORD *)(v34[8] + 128LL), a1, (__int64)v8, (__int64 *)(a2 + 48), 0, &v33, v37);
              if ( v33 || !(unsigned int)sub_8AE140((__int64 *)&v36, 0, 0) )
              {
                v11 = 0;
                sub_89A1C0((__int64 *)&v34, &v35);
                return v11;
              }
              v22 = v34;
              v24 = (__int64)v36;
              v14 = v35;
              v23 = *((_BYTE *)v34 + 57) & 8;
            }
            while ( *(_BYTE *)(v24 + 140) == 12 )
              v24 = *(_QWORD *)(v24 + 160);
            v36 = (__int64 **)v24;
            v20 = v14;
            if ( (v14[3] & 1) != 0 )
            {
              if ( (_BYTE)v23 )
                goto LABEL_52;
              if ( !(unsigned int)sub_8D2780(v24) )
              {
                *((_BYTE *)v35 + 25) &= ~1u;
                sub_89A1C0((__int64 *)&v34, &v35);
                return 0;
              }
              v29 = sub_724D80(1);
              v30 = (__int64)v36;
              if ( *((_BYTE *)v36 + 140) == 12 )
              {
                do
                  v30 = *(_QWORD *)(v30 + 160);
                while ( *(_BYTE *)(v30 + 140) == 12 );
              }
              v31 = v29;
              sub_72BBE0((__int64)v29, v35[4], *(_BYTE *)(v30 + 160));
              v20 = v35;
              *((_BYTE *)v35 + 24) &= ~1u;
              v20[4] = (__int64)v31;
            }
            else
            {
              v25 = v14[4];
              if ( (_BYTE)v23 )
              {
                v28 = sub_696F90(*(_QWORD *)(v22[8] + 128LL), v25, 0, 0, 0, (__int64)a1, (__int64)v8);
                v20 = v35;
                v11 = v28;
              }
              else if ( (v22[9] & 1) != 0 )
              {
                v26 = *(_QWORD *)(v25 + 128);
                v11 = 1;
                if ( v26 != v24 )
                {
                  v27 = sub_8D97D0(v24, v26, 32, v23, v22);
                  v20 = v35;
                  v11 = v27 != 0;
                }
              }
            }
            goto LABEL_27;
          }
LABEL_45:
          sub_89A1C0((__int64 *)&v34, &v35);
          v14 = v35;
          if ( !v35 )
            return v11;
        }
        if ( (v14[3] & 1) == 0 && !v14[4] )
          goto LABEL_45;
        goto LABEL_38;
      }
      return v11;
  }
}
