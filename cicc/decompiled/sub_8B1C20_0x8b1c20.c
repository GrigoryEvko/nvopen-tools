// Function: sub_8B1C20
// Address: 0x8b1c20
//
__int64 **__fastcall sub_8B1C20(__int64 a1, __int64 a2, __m128i **a3, _QWORD *a4, unsigned int a5)
{
  __int64 v5; // r9
  __m128i *v7; // r12
  __int64 v8; // rbx
  char v9; // al
  __int64 v10; // r13
  __int64 **v11; // r15
  __m128i *v14; // rax
  unsigned int v15; // r14d
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 i; // r15
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 *v23; // r9
  int v24; // r8d
  char v25; // al
  __int64 *v26; // rax
  char v27; // dl
  int v28; // eax
  char v29; // al
  char v30; // dl
  _QWORD *v31; // rdi
  __int64 v32; // rax
  unsigned int v33; // r14d
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 *v37; // [rsp+10h] [rbp-D0h]
  _QWORD *v38; // [rsp+20h] [rbp-C0h]
  __int64 v40; // [rsp+28h] [rbp-B8h]
  __int64 v41; // [rsp+28h] [rbp-B8h]
  int v42; // [rsp+3Ch] [rbp-A4h] BYREF
  __int64 *v43; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD *v44; // [rsp+48h] [rbp-98h] BYREF
  __m128i v45[5]; // [rsp+50h] [rbp-90h] BYREF
  int v46; // [rsp+A0h] [rbp-40h]

  v5 = (__int64)a3;
  v7 = (__m128i *)a2;
  v8 = a1;
  v9 = *(_BYTE *)(a1 + 80);
  v38 = a4;
  v42 = 0;
  switch ( v9 )
  {
    case 4:
    case 5:
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      goto LABEL_3;
    case 6:
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      goto LABEL_3;
    case 9:
    case 10:
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      if ( !a4 )
        goto LABEL_9;
      goto LABEL_4;
    case 19:
    case 20:
    case 21:
    case 22:
      v10 = *(_QWORD *)(a1 + 88);
      goto LABEL_3;
    default:
      v10 = 0;
LABEL_3:
      if ( !a4 )
LABEL_9:
        v38 = **(_QWORD ***)(v10 + 328);
LABEL_4:
      if ( (unsigned __int64)*(unsigned int *)(v10 + 392) <= unk_4D042F0 )
      {
        if ( a3 )
        {
          v14 = sub_8A3C00((__int64)v38, a2, 0, (__int64 *)(a1 + 48));
          v5 = (__int64)a3;
          v15 = a5 | 0x4000;
          v7 = v14;
          *a3 = v14;
          if ( (*(_BYTE *)(v10 + 160) & 0x10) != 0 && (a5 & 0x20000) != 0 )
          {
            v33 = a5;
            LODWORD(v11) = 1;
            v15 = v33 | 0x4140;
            if ( !v14 )
              return 0;
LABEL_15:
            if ( !dword_4D04494 || (_DWORD)qword_4F077B4 && qword_4F077A0 || (v15 & 8) != 0 || v5 )
              goto LABEL_18;
            v25 = *(_BYTE *)(a1 + 80);
            if ( v25 == 16 )
            {
              a1 = **(_QWORD **)(a1 + 88);
              v25 = *(_BYTE *)(a1 + 80);
            }
            if ( v25 == 24 )
            {
              a1 = *(_QWORD *)(a1 + 88);
              v25 = *(_BYTE *)(a1 + 80);
            }
            if ( (unsigned __int8)(v25 - 10) <= 1u )
            {
              v35 = *(_QWORD *)(a1 + 88);
              if ( (*(_BYTE *)(v35 + 194) & 0x40) == 0 )
                goto LABEL_33;
              do
                v35 = *(_QWORD *)(v35 + 232);
              while ( (*(_BYTE *)(v35 + 194) & 0x40) != 0 );
            }
            else
            {
              if ( v25 != 20 || (v36 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 176LL), (*(_BYTE *)(v36 + 194) & 0x40) == 0) )
              {
LABEL_33:
                if ( !(unsigned int)sub_8A00C0(a1, v7->m128i_i64, 0) )
                  return 0;
LABEL_18:
                v41 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                sub_892150(v45);
                v17 = *(_QWORD *)(v10 + 176);
                v46 = (int)v11;
                for ( i = *(_QWORD *)(v17 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
                  ;
                ++*(_DWORD *)(v10 + 392);
                v37 = (__int64 *)(v8 + 48);
                if ( *(_BYTE *)(v41 + 4) == 14 && *(_QWORD *)(v41 + 368) == v8 )
                {
                  v11 = sub_8A2270(i, v7, (__int64)v38, v37, v15, &v42, v45);
                }
                else
                {
                  sub_8600D0(0xEu, -1, 0, v17);
                  v19 = i;
                  *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 368) = v8;
                  v11 = sub_8A2270(i, v7, (__int64)v38, v37, v15, &v42, v45);
                  sub_863FC0(v19, (__int64)v7, v20, v21, v22, v23);
                }
                v24 = v42;
                --*(_DWORD *)(v10 + 392);
                if ( !v24 )
                {
                  sub_892150(v45);
                  sub_89A1A0(v38, v7->m128i_i64, &v44, &v43);
                  v26 = v43;
                  if ( v43 )
                  {
                    v27 = *((_BYTE *)v43 + 8);
                    if ( v27 )
                      goto LABEL_50;
LABEL_37:
                    if ( v26[4] )
                    {
                      while ( 1 )
                      {
                        sub_89A1C0((__int64 *)&v44, &v43);
                        v26 = v43;
                        if ( !v43 )
                          break;
                        v27 = *((_BYTE *)v43 + 8);
                        if ( !v27 )
                          goto LABEL_37;
LABEL_50:
                        if ( v27 == 1 )
                        {
                          if ( (v26[3] & 1) == 0 && !v26[4] )
                            break;
                        }
                        else
                        {
                          if ( !v26[4] )
                            break;
                          if ( v27 == 2 && (*(_BYTE *)(v44[8] + 266LL) & 4) != 0 )
                          {
                            v30 = *((_BYTE *)v26 + 24);
                            if ( (v30 & 4) == 0 )
                            {
                              v31 = (_QWORD *)v26[4];
                              *((_BYTE *)v26 + 24) = v30 | 4;
                              v32 = sub_8794A0(v31);
                              if ( !(unsigned int)sub_8B44F0(
                                                    **(_QWORD **)(v32 + 32),
                                                    **(_QWORD **)(v44[8] + 32LL),
                                                    v7,
                                                    v38,
                                                    v37,
                                                    &v42,
                                                    v45) )
                                return 0;
                            }
                          }
                        }
                      }
                    }
                    v28 = v42;
                  }
                  else
                  {
                    v28 = v42;
                  }
                  if ( !v28 )
                  {
                    if ( !unk_4D04484 )
                    {
                      if ( !v11 )
                        return 0;
LABEL_45:
                      sub_8DCB20(v11);
                      return v11;
                    }
                    if ( v11 )
                    {
                      v29 = *(_BYTE *)(*(_QWORD *)(v10 + 176) + 174LL);
                      if ( v29 == 3 )
                      {
                        v34 = *(_QWORD **)(v8 + 64);
                        if ( (*(_BYTE *)(v34[21] + 109LL) & 0x20) != 0 )
                          sub_8B1B50(v34, v7, (__int64)v11);
                      }
                      else if ( v29 == 6 )
                      {
                        sub_8C0170(*(_QWORD *)(v8 + 64), v7, v11);
                      }
                      goto LABEL_45;
                    }
                  }
                }
                return 0;
              }
              do
                v36 = *(_QWORD *)(v36 + 232);
              while ( (*(_BYTE *)(v36 + 194) & 0x40) != 0 );
              v35 = *(_QWORD *)(v36 + 248);
            }
            a1 = *(_QWORD *)v35;
            goto LABEL_33;
          }
          if ( !v14 )
            return 0;
        }
        else
        {
          if ( !a2 )
            return 0;
          v15 = a5;
        }
        v40 = v5;
        v16 = sub_894B30(a1, v10, v7, v15, 0, v5);
        v5 = v40;
        v11 = (__int64 **)v16;
        if ( v16 )
          return v11;
        goto LABEL_15;
      }
      if ( a3 )
      {
        *a3 = 0;
        v11 = 0;
        sub_861C90();
      }
      else
      {
        sub_861C90();
        return 0;
      }
      return v11;
  }
}
