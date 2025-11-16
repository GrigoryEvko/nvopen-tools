// Function: sub_3863850
// Address: 0x3863850
//
__int64 __fastcall sub_3863850(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // r12
  __int64 v11; // r13
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r15
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // esi
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  unsigned int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int64 v33; // rcx
  _QWORD *v34; // rax
  unsigned __int64 v35; // rax
  unsigned int v36; // esi
  int v37; // eax
  __int64 v38; // rax
  unsigned int v39; // esi
  int v40; // eax
  __int64 v41; // rax
  unsigned __int64 v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // rax
  __int64 *v45; // rsi
  unsigned int v46; // edi
  _QWORD *v47; // rcx
  int v48; // r9d
  __int64 *v49; // r11
  int v50; // edx
  int v51; // edi
  __int64 v52; // [rsp-68h] [rbp-68h]
  __int64 v53; // [rsp-60h] [rbp-60h]
  __int64 v54; // [rsp-60h] [rbp-60h]
  unsigned __int64 v55; // [rsp-60h] [rbp-60h]
  __int64 v56; // [rsp-60h] [rbp-60h]
  __int64 v57; // [rsp-58h] [rbp-58h]
  __int64 v58; // [rsp-58h] [rbp-58h]
  unsigned __int64 v59; // [rsp-58h] [rbp-58h]
  __int64 v60; // [rsp-58h] [rbp-58h]
  unsigned __int64 v61; // [rsp-58h] [rbp-58h]
  __int64 v62; // [rsp-58h] [rbp-58h]
  unsigned __int64 v63; // [rsp-58h] [rbp-58h]
  __int64 v64; // [rsp-50h] [rbp-50h]
  unsigned __int64 v65; // [rsp-50h] [rbp-50h]
  unsigned __int64 v66; // [rsp-50h] [rbp-50h]
  __int64 v67; // [rsp-50h] [rbp-50h]
  unsigned __int64 v68; // [rsp-50h] [rbp-50h]
  __int64 v69; // [rsp-48h] [rbp-48h] BYREF
  __int64 *v70; // [rsp-40h] [rbp-40h] BYREF

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result <= 0x17u )
    return result;
  result = (unsigned int)(result - 54);
  if ( (unsigned __int8)result > 1u )
    return result;
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD *)(a2 - 24);
  v8 = *(_QWORD *)(a1 + 24);
  v9 = *(_QWORD *)(v6 + 112);
  v69 = v7;
  result = sub_14C4510(v7, v9, v8);
  v10 = result;
  if ( !result )
    return result;
  v11 = (__int64)sub_1494E70(*(_QWORD *)a1, result, a3, a4);
  v12 = sub_1495DC0(*(_QWORD *)a1, a3, a4);
  v13 = sub_157EB90(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 32LL));
  v14 = sub_1632FA0(v13);
  v64 = sub_1456040(v11);
  v15 = sub_15A9FE0(v14, v64);
  v16 = v64;
  v17 = 1;
  v18 = v15;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v16 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v38 = *(_QWORD *)(v16 + 32);
        v16 = *(_QWORD *)(v16 + 24);
        v17 *= v38;
        continue;
      case 1:
        v29 = 16;
        goto LABEL_18;
      case 2:
        v29 = 32;
        goto LABEL_18;
      case 3:
      case 9:
        v29 = 64;
        goto LABEL_18;
      case 4:
        v29 = 80;
        goto LABEL_18;
      case 5:
      case 6:
        v29 = 128;
        goto LABEL_18;
      case 7:
        v60 = v17;
        v36 = 0;
        v68 = v18;
        goto LABEL_27;
      case 0xB:
        v29 = *(_DWORD *)(v16 + 8) >> 8;
        goto LABEL_18;
      case 0xD:
        v58 = v17;
        v66 = v18;
        v34 = (_QWORD *)sub_15A9930(v14, v16);
        v18 = v66;
        v17 = v58;
        v29 = 8LL * *v34;
        goto LABEL_18;
      case 0xE:
        v53 = v17;
        v59 = v18;
        v67 = *(_QWORD *)(v16 + 32);
        v35 = sub_12BE0A0(v14, *(_QWORD *)(v16 + 24));
        v18 = v59;
        v17 = v53;
        v29 = 8 * v67 * v35;
        goto LABEL_18;
      case 0xF:
        v60 = v17;
        v68 = v18;
        v36 = *(_DWORD *)(v16 + 8) >> 8;
LABEL_27:
        v37 = sub_15A9520(v14, v36);
        v18 = v68;
        v17 = v60;
        v29 = (unsigned int)(8 * v37);
LABEL_18:
        v65 = v18 * ((v18 + ((unsigned __int64)(v29 * v17 + 7) >> 3) - 1) / v18);
        v57 = sub_1456040(v12);
        v30 = sub_15A9FE0(v14, v57);
        v31 = v57;
        v32 = 1;
        v33 = v30;
        break;
    }
    break;
  }
  while ( 2 )
  {
    switch ( *(_BYTE *)(v31 + 8) )
    {
      case 1:
        v19 = 16;
        goto LABEL_8;
      case 2:
        v19 = 32;
        goto LABEL_8;
      case 3:
      case 9:
        v19 = 64;
        goto LABEL_8;
      case 4:
        v19 = 80;
        goto LABEL_8;
      case 5:
      case 6:
        v19 = 128;
        goto LABEL_8;
      case 7:
        v54 = v32;
        v39 = 0;
        v61 = v33;
        goto LABEL_36;
      case 0xB:
        v19 = *(_DWORD *)(v31 + 8) >> 8;
        goto LABEL_8;
      case 0xD:
        v56 = v32;
        v63 = v33;
        v43 = (_QWORD *)sub_15A9930(v14, v31);
        v33 = v63;
        v32 = v56;
        v19 = 8LL * *v43;
        goto LABEL_8;
      case 0xE:
        v52 = v32;
        v55 = v33;
        v62 = *(_QWORD *)(v31 + 32);
        v42 = sub_12BE0A0(v14, *(_QWORD *)(v31 + 24));
        v33 = v55;
        v32 = v52;
        v19 = 8 * v62 * v42;
        goto LABEL_8;
      case 0xF:
        v54 = v32;
        v61 = v33;
        v39 = *(_DWORD *)(v31 + 8) >> 8;
LABEL_36:
        v40 = sub_15A9520(v14, v39);
        v33 = v61;
        v32 = v54;
        v19 = (unsigned int)(8 * v40);
LABEL_8:
        v20 = *(_QWORD *)(*(_QWORD *)a1 + 112LL);
        if ( v33 * ((v33 + ((unsigned __int64)(v19 * v32 + 7) >> 3) - 1) / v33) < v65 )
        {
          v44 = sub_1456040(v11);
          v12 = sub_14747F0(v20, v12, v44, 0);
        }
        else
        {
          v21 = sub_1456040(v12);
          v11 = sub_147BE00(v20, v11, v21);
        }
        v22 = sub_14806B0(v20, v11, v12, 0, 0);
        result = sub_1477C30(v20, v22);
        if ( (_BYTE)result )
          return result;
        v23 = *(_DWORD *)(a1 + 88);
        if ( !v23 )
        {
          ++*(_QWORD *)(a1 + 64);
          goto LABEL_64;
        }
        v24 = v69;
        v25 = *(_QWORD *)(a1 + 72);
        v26 = (v23 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
        v27 = (__int64 *)(v25 + 16 * v26);
        v28 = *v27;
        if ( v69 == *v27 )
          goto LABEL_13;
        v48 = 1;
        v49 = 0;
        while ( 1 )
        {
          if ( v28 == -8 )
          {
            v50 = *(_DWORD *)(a1 + 80);
            if ( v49 )
              v27 = v49;
            ++*(_QWORD *)(a1 + 64);
            v51 = v50 + 1;
            if ( 4 * (v50 + 1) < 3 * v23 )
            {
              if ( v23 - *(_DWORD *)(a1 + 84) - v51 > v23 >> 3 )
              {
LABEL_60:
                *(_DWORD *)(a1 + 80) = v51;
                if ( *v27 != -8 )
                  --*(_DWORD *)(a1 + 84);
                *v27 = v24;
                v27[1] = 0;
                break;
              }
LABEL_65:
              sub_14669A0(a1 + 64, v23);
              sub_145CB40(a1 + 64, &v69, &v70);
              v27 = v70;
              v24 = v69;
              v51 = *(_DWORD *)(a1 + 80) + 1;
              goto LABEL_60;
            }
LABEL_64:
            v23 *= 2;
            goto LABEL_65;
          }
          if ( !v49 && v28 == -16 )
            v49 = v27;
          LODWORD(v26) = (v23 - 1) & (v48 + v26);
          v27 = (__int64 *)(v25 + 16LL * (unsigned int)v26);
          v28 = *v27;
          if ( v69 == *v27 )
            break;
          ++v48;
        }
LABEL_13:
        v27[1] = v10;
        result = *(_QWORD *)(a1 + 104);
        if ( *(_QWORD *)(a1 + 112) != result )
          return (__int64)sub_16CCBA0(a1 + 96, v10);
        v45 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 124));
        v46 = *(_DWORD *)(a1 + 124);
        if ( (__int64 *)result == v45 )
        {
LABEL_52:
          if ( v46 >= *(_DWORD *)(a1 + 120) )
          {
            return (__int64)sub_16CCBA0(a1 + 96, v10);
          }
          else
          {
            *(_DWORD *)(a1 + 124) = v46 + 1;
            *v45 = v10;
            ++*(_QWORD *)(a1 + 96);
          }
        }
        else
        {
          v47 = 0;
          while ( v10 != *(_QWORD *)result )
          {
            if ( *(_QWORD *)result == -2 )
              v47 = (_QWORD *)result;
            result += 8;
            if ( v45 == (__int64 *)result )
            {
              if ( !v47 )
                goto LABEL_52;
              *v47 = v10;
              --*(_DWORD *)(a1 + 128);
              ++*(_QWORD *)(a1 + 96);
              return result;
            }
          }
        }
        return result;
      case 0x10:
        v41 = *(_QWORD *)(v31 + 32);
        v31 = *(_QWORD *)(v31 + 24);
        v32 *= v41;
        continue;
      default:
        BUG();
    }
  }
}
