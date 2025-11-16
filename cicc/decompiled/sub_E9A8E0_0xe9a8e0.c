// Function: sub_E9A8E0
// Address: 0xe9a8e0
//
void __fastcall sub_E9A8E0(__int64 a1, _DWORD *a2, _QWORD *a3, _DWORD *a4, _QWORD *a5)
{
  unsigned int v5; // r15d
  unsigned int v7; // ecx
  __int64 v10; // rsi
  __int128 v11; // rax
  signed __int64 v12; // r9
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r10
  int v15; // eax
  unsigned int v16; // eax
  void (*v17)(); // r11
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // esi
  unsigned int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // r8
  void (*v24)(); // r11
  unsigned int v25; // esi
  unsigned int v26; // ecx
  void (*v27)(); // r11
  unsigned int v28; // edx
  __int128 v29; // rax
  __int128 v30; // rax
  __int128 v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // eax
  unsigned int v34; // eax
  __int64 v35; // r9
  void (*v36)(); // r11
  char v37; // [rsp+Bh] [rbp-75h]
  signed __int64 v38; // [rsp+10h] [rbp-70h]
  char v39; // [rsp+13h] [rbp-6Dh]
  char v40; // [rsp+17h] [rbp-69h]
  bool v41; // [rsp+18h] [rbp-68h]
  unsigned __int64 v42; // [rsp+18h] [rbp-68h]
  signed __int64 v43; // [rsp+18h] [rbp-68h]
  unsigned int v44; // [rsp+20h] [rbp-60h]
  unsigned int v45; // [rsp+20h] [rbp-60h]
  unsigned int v46; // [rsp+20h] [rbp-60h]
  unsigned int v47; // [rsp+24h] [rbp-5Ch]
  unsigned int v48; // [rsp+24h] [rbp-5Ch]
  __int128 v50; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v51[8]; // [rsp+40h] [rbp-40h] BYREF

  v5 = a2[13];
  if ( v5 == 5 )
  {
    v7 = a2[11];
    if ( v7 <= 0x1F )
    {
      v41 = ((0xD8000222uLL >> v7) & 1) == 0;
      if ( ((0xD8000222uLL >> v7) & 1) != 0 )
      {
        if ( (unsigned int)sub_CC78E0((__int64)a2) )
        {
          v50 = 0;
          switch ( a2[11] )
          {
            case 1:
            case 9:
              sub_CC7B70((__int64)a2, (__int64)&v50);
              break;
            case 5:
            case 0x1B:
              *(_QWORD *)&v11 = sub_CC7C70((__int64)a2);
              v50 = v11;
              break;
            case 0x1C:
              *(_QWORD *)&v31 = sub_CC7D30((__int64)a2);
              v50 = v31;
              break;
            case 0x1E:
              *(_QWORD *)&v30 = sub_CC7D70((__int64)a2);
              v50 = v30;
              break;
            case 0x1F:
              *(_QWORD *)&v29 = sub_CC78E0((__int64)a2);
              v50 = v29;
              break;
            default:
LABEL_54:
              BUG();
          }
          v12 = sub_E985A0(a2, v50, *((__int64 *)&v50 + 1));
          v14 = v13;
          v47 = HIDWORD(v12) & 0x7FFFFFFF;
          v44 = v13 & 0x7FFFFFFF;
          v15 = a2[11];
          switch ( v15 )
          {
            case 1:
            case 9:
              v5 = 10;
              v28 = 14;
              goto LABEL_33;
            case 5:
              if ( a2[12] != 32 )
                goto LABEL_42;
              goto LABEL_14;
            case 27:
LABEL_42:
              v5 = 12;
              v28 = 0;
              goto LABEL_33;
            case 28:
              v28 = 0;
LABEL_33:
              if ( (unsigned int)v12 >= v5 && ((_DWORD)v12 != v5 || v28 <= v47) )
                goto LABEL_14;
              if ( !a4 || (v15 & 0xFFFFFFF7) != 1 )
                goto LABEL_26;
              if ( a4[12] == 32 )
                goto LABEL_22;
              goto LABEL_25;
            case 30:
            case 31:
LABEL_14:
              if ( a2[12] == 32 && a4 && (a4[11] & 0xFFFFFFF7) == 1 )
              {
                v40 = HIBYTE(v12);
                v42 = v14 >> 24;
                v51[0] = 0;
                v51[1] = 0;
                sub_E9A8E0(a1, a4, a5, 0, v51);
                v32 = 0;
                v33 = v44;
                if ( (v42 & 0x80u) == 0LL )
                  v33 = 0;
                v46 = v33;
                if ( v40 < 0 )
                  v32 = v47;
                v48 = v32;
                v34 = sub_E97A00((__int64)a2);
                if ( v36 != nullsub_104 )
                  ((void (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, __int64, _QWORD, _QWORD))v36)(
                    a1,
                    v34,
                    (unsigned int)v35,
                    v48,
                    v46,
                    v35,
                    *a3,
                    a3[1]);
              }
              else
              {
                v16 = sub_E97A00((__int64)a2);
                if ( v17 != nullsub_103 )
                {
                  v39 = BYTE3(v14);
                  v43 = v12;
                  ((void (__fastcall *)(__int64, _QWORD, _QWORD))v17)(a1, v16, (unsigned int)v12);
                  v12 = v43;
                  BYTE3(v14) = v39;
                }
                if ( a4 && (a2[11] & 0xFFFFFFF7) == 1 && a4[12] == 32 )
                {
                  v41 = 1;
LABEL_22:
                  v37 = BYTE3(v14);
                  v38 = v12;
                  v18 = sub_CC7C70((__int64)a4);
                  v20 = sub_E985A0(a4, v18, v19);
                  v21 = sub_E97A00((__int64)a4);
                  v12 = v38;
                  BYTE3(v14) = v37;
                  if ( v24 != nullsub_104 )
                  {
                    ((void (__fastcall *)(__int64, _QWORD, _QWORD, __int64, __int64, signed __int64, _QWORD))v24)(
                      a1,
                      v21,
                      v20,
                      v22,
                      v23,
                      v38,
                      *a5);
                    BYTE3(v14) = v37;
                    v12 = v38;
                  }
                  if ( !v41 )
                  {
LABEL_25:
                    v15 = a2[11];
LABEL_26:
                    v25 = v44;
                    v26 = 0;
                    v27 = *(void (**)())(*(_QWORD *)a1 + 248LL);
                    if ( (v14 & 0x80000000) == 0 )
                      v25 = 0;
                    if ( v12 < 0 )
                      v26 = v47;
                    v45 = v25;
                    switch ( v15 )
                    {
                      case 1:
                      case 9:
                        v10 = 1;
                        goto LABEL_7;
                      case 5:
                        v10 = 0;
                        goto LABEL_7;
                      case 27:
                        v10 = 2;
                        goto LABEL_7;
                      case 28:
                        v10 = 3;
LABEL_7:
                        if ( v27 != nullsub_102 )
                          ((void (__fastcall *)(__int64, __int64, _QWORD, _QWORD, _QWORD, signed __int64, _QWORD, _QWORD))v27)(
                            a1,
                            v10,
                            (unsigned int)v12,
                            v26,
                            v45,
                            v12,
                            *a3,
                            a3[1]);
                        return;
                      default:
                        goto LABEL_54;
                    }
                  }
                }
              }
              break;
            default:
              goto LABEL_54;
          }
        }
      }
    }
  }
}
