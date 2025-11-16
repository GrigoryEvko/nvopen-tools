// Function: sub_1B70620
// Address: 0x1b70620
//
__int64 __fastcall sub_1B70620(__int64 ***a1, int a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  _BYTE *v7; // r8
  __int64 v11; // r12
  __int64 **v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rbx
  unsigned __int64 v16; // r12
  __int64 v17; // rax
  __int64 *v19; // rbx
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rcx
  unsigned __int64 v23; // r9
  unsigned int v24; // esi
  int v25; // eax
  __int64 *v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rax
  unsigned int v29; // esi
  int v30; // eax
  unsigned int v31; // eax
  __int64 v32; // rsi
  __int64 v33; // rcx
  unsigned __int64 v34; // r10
  _QWORD *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // esi
  int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // rax
  unsigned int v45; // esi
  int v46; // eax
  __int64 v47; // rdi
  __int64 v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // [rsp+8h] [rbp-68h]
  __int64 v51; // [rsp+10h] [rbp-60h]
  unsigned __int64 v52; // [rsp+10h] [rbp-60h]
  unsigned __int64 v53; // [rsp+18h] [rbp-58h]
  __int64 v54; // [rsp+18h] [rbp-58h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  unsigned __int64 v57; // [rsp+20h] [rbp-50h]
  __int64 v58; // [rsp+20h] [rbp-50h]
  __int64 v59; // [rsp+28h] [rbp-48h]
  __int64 v60; // [rsp+28h] [rbp-48h]
  unsigned __int64 v61; // [rsp+28h] [rbp-48h]
  __int64 v62; // [rsp+28h] [rbp-48h]
  unsigned __int64 v63; // [rsp+28h] [rbp-48h]
  __int64 v64; // [rsp+28h] [rbp-48h]
  unsigned __int64 v65; // [rsp+28h] [rbp-48h]
  __int64 v66; // [rsp+30h] [rbp-40h]
  __int64 v67; // [rsp+30h] [rbp-40h]
  unsigned __int64 v68; // [rsp+30h] [rbp-40h]
  __int64 v69; // [rsp+30h] [rbp-40h]
  unsigned __int64 v70; // [rsp+30h] [rbp-40h]
  __int64 v71; // [rsp+30h] [rbp-40h]
  _BYTE *v75; // [rsp+38h] [rbp-38h]
  __int64 v76; // [rsp+38h] [rbp-38h]
  _BYTE *v77; // [rsp+38h] [rbp-38h]
  __int64 v78; // [rsp+38h] [rbp-38h]

  v7 = (_BYTE *)a4;
  v11 = 1;
  v12 = *a1;
  while ( 2 )
  {
    switch ( *((_BYTE *)v12 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v26 = v12[4];
        v12 = (__int64 **)v12[3];
        v11 *= (_QWORD)v26;
        continue;
      case 1:
        v13 = 16;
        goto LABEL_5;
      case 2:
        v13 = 32;
        goto LABEL_5;
      case 3:
      case 9:
        v13 = 64;
        goto LABEL_5;
      case 4:
        v13 = 80;
        goto LABEL_5;
      case 5:
      case 6:
        v13 = 128;
        goto LABEL_5;
      case 7:
        v24 = 0;
        goto LABEL_15;
      case 0xB:
        v13 = *((_DWORD *)v12 + 2) >> 8;
        goto LABEL_5;
      case 0xD:
        v27 = (_QWORD *)sub_15A9930(a4, (__int64)v12);
        v7 = (_BYTE *)a4;
        v13 = 8LL * *v27;
        goto LABEL_5;
      case 0xE:
        v19 = v12[4];
        v66 = (__int64)v12[3];
        v20 = sub_15A9FE0(a4, v66);
        v21 = v66;
        v7 = (_BYTE *)a4;
        v22 = 1;
        v23 = v20;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v21 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v38 = *(_QWORD *)(v21 + 32);
              v21 = *(_QWORD *)(v21 + 24);
              v22 *= v38;
              continue;
            case 1:
              v36 = 16;
              goto LABEL_36;
            case 2:
              v36 = 32;
              goto LABEL_36;
            case 3:
            case 9:
              v36 = 64;
              goto LABEL_36;
            case 4:
              v36 = 80;
              goto LABEL_36;
            case 5:
            case 6:
              v36 = 128;
              goto LABEL_36;
            case 7:
              v60 = v22;
              v39 = 0;
              v68 = v23;
              goto LABEL_46;
            case 0xB:
              v36 = *(_DWORD *)(v21 + 8) >> 8;
              goto LABEL_36;
            case 0xD:
              v62 = v22;
              v70 = v23;
              v43 = (_QWORD *)sub_15A9930(a4, v21);
              v7 = (_BYTE *)a4;
              v23 = v70;
              v22 = v62;
              v36 = 8LL * *v43;
              goto LABEL_36;
            case 0xE:
              v41 = a4;
              v51 = v22;
              v53 = v23;
              v55 = *(_QWORD *)(v21 + 24);
              v69 = a4;
              v78 = *(_QWORD *)(v21 + 32);
              v61 = (unsigned int)sub_15A9FE0(v41, v55);
              v42 = sub_127FA20(v69, v55);
              v7 = (_BYTE *)v69;
              v23 = v53;
              v22 = v51;
              v36 = 8 * v78 * v61 * ((v61 + ((unsigned __int64)(v42 + 7) >> 3) - 1) / v61);
              goto LABEL_36;
            case 0xF:
              v60 = v22;
              v68 = v23;
              v39 = *(_DWORD *)(v21 + 8) >> 8;
LABEL_46:
              v40 = sub_15A9520(a4, v39);
              v7 = (_BYTE *)a4;
              v23 = v68;
              v22 = v60;
              v36 = (unsigned int)(8 * v40);
LABEL_36:
              v13 = 8 * v23 * (_QWORD)v19 * ((v23 + ((unsigned __int64)(v36 * v22 + 7) >> 3) - 1) / v23);
              break;
          }
          goto LABEL_5;
        }
      case 0xF:
        v24 = *((_DWORD *)v12 + 2) >> 8;
LABEL_15:
        v25 = sub_15A9520(a4, v24);
        v7 = (_BYTE *)a4;
        v13 = (unsigned int)(8 * v25);
LABEL_5:
        v14 = a3;
        v15 = 1;
        v16 = (unsigned __int64)(v13 * v11 + 7) >> 3;
        while ( 1 )
        {
          switch ( *(_BYTE *)(v14 + 8) )
          {
            case 1:
              v17 = 16;
              goto LABEL_9;
            case 2:
              v17 = 32;
              goto LABEL_9;
            case 3:
            case 9:
              v17 = 64;
              goto LABEL_9;
            case 4:
              v17 = 80;
              goto LABEL_9;
            case 5:
            case 6:
              v17 = 128;
              goto LABEL_9;
            case 7:
              v29 = 0;
              goto LABEL_27;
            case 0xB:
              v17 = *(_DWORD *)(v14 + 8) >> 8;
              goto LABEL_9;
            case 0xD:
              v77 = v7;
              v35 = (_QWORD *)sub_15A9930((__int64)v7, v14);
              v7 = v77;
              v17 = 8LL * *v35;
              goto LABEL_9;
            case 0xE:
              v67 = (__int64)v7;
              v59 = *(_QWORD *)(v14 + 24);
              v76 = *(_QWORD *)(v14 + 32);
              v31 = sub_15A9FE0((__int64)v7, v59);
              v32 = v59;
              v7 = (_BYTE *)v67;
              v33 = 1;
              v34 = v31;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v32 + 8) )
                {
                  case 1:
                    v37 = 16;
                    goto LABEL_39;
                  case 2:
                    v37 = 32;
                    goto LABEL_39;
                  case 3:
                  case 9:
                    v37 = 64;
                    goto LABEL_39;
                  case 4:
                    v37 = 80;
                    goto LABEL_39;
                  case 5:
                  case 6:
                    v37 = 128;
                    goto LABEL_39;
                  case 7:
                    v56 = v33;
                    v45 = 0;
                    v63 = v34;
                    goto LABEL_55;
                  case 0xB:
                    v37 = *(_DWORD *)(v32 + 8) >> 8;
                    goto LABEL_39;
                  case 0xD:
                    v58 = v33;
                    v65 = v34;
                    v49 = (_QWORD *)sub_15A9930(v67, v32);
                    v7 = (_BYTE *)v67;
                    v34 = v65;
                    v33 = v58;
                    v37 = 8LL * *v49;
                    goto LABEL_39;
                  case 0xE:
                    v47 = v67;
                    v50 = v33;
                    v52 = v34;
                    v54 = *(_QWORD *)(v32 + 24);
                    v64 = v67;
                    v71 = *(_QWORD *)(v32 + 32);
                    v57 = (unsigned int)sub_15A9FE0(v47, v54);
                    v48 = sub_127FA20(v64, v54);
                    v7 = (_BYTE *)v64;
                    v34 = v52;
                    v33 = v50;
                    v37 = 8 * v71 * v57 * ((v57 + ((unsigned __int64)(v48 + 7) >> 3) - 1) / v57);
                    goto LABEL_39;
                  case 0xF:
                    v56 = v33;
                    v63 = v34;
                    v45 = *(_DWORD *)(v32 + 8) >> 8;
LABEL_55:
                    v46 = sub_15A9520(v67, v45);
                    v7 = (_BYTE *)v67;
                    v34 = v63;
                    v33 = v56;
                    v37 = (unsigned int)(8 * v46);
LABEL_39:
                    v17 = 8 * v76 * v34 * ((v34 + ((unsigned __int64)(v37 * v33 + 7) >> 3) - 1) / v34);
                    goto LABEL_9;
                  case 0x10:
                    v44 = *(_QWORD *)(v32 + 32);
                    v32 = *(_QWORD *)(v32 + 24);
                    v33 *= v44;
                    continue;
                  default:
                    goto LABEL_3;
                }
              }
            case 0xF:
              v29 = *(_DWORD *)(v14 + 8) >> 8;
LABEL_27:
              v75 = v7;
              v30 = sub_15A9520((__int64)v7, v29);
              v7 = v75;
              v17 = (unsigned int)(8 * v30);
LABEL_9:
              if ( a2 + (unsigned int)((unsigned __int64)(v17 * v15 + 7) >> 3) > (unsigned int)v16 )
                return 0;
              else
                return sub_1B6FF90(a1, a2, a3, v7, a5, a6, a7);
            case 0x10:
              v28 = *(_QWORD *)(v14 + 32);
              v14 = *(_QWORD *)(v14 + 24);
              v15 *= v28;
              break;
            default:
LABEL_3:
              BUG();
          }
        }
    }
  }
}
