// Function: sub_33E2C00
// Address: 0x33e2c00
//
__int64 __fastcall sub_33E2C00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // r13d
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  int v19; // r13d
  int v20; // r14d
  int v21; // r13d
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r12
  int v31; // r13d
  unsigned __int64 v32; // r12
  unsigned __int64 v33; // rcx
  int v34; // eax
  char v35; // si
  unsigned __int64 v36; // rcx
  char v37; // di
  unsigned __int64 v38; // rsi
  unsigned __int64 v39; // r8
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rsi
  unsigned __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  unsigned __int64 v52; // rcx
  unsigned __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  unsigned __int16 *v58; // rax
  unsigned __int16 v59; // r13
  __int64 v60; // rax
  int *v61; // r13
  int *v62; // r14
  int v63; // r15d
  unsigned __int16 v64; // [rsp+10h] [rbp-40h] BYREF
  __int64 v65; // [rsp+18h] [rbp-38h]

  result = *(unsigned int *)(a2 + 24);
  if ( (int)result > 372 )
  {
    result = (unsigned int)(result - 465);
    v9 = *(unsigned __int16 *)(a2 + 96);
    switch ( (int)result )
    {
      case 0:
      case 2:
      case 3:
      case 5:
LABEL_7:
        if ( !(_WORD)v9 )
          v9 = *(_QWORD *)(a2 + 104);
        sub_D953B0(a1, v9, a3, a4, a5, a6);
        sub_9C8C60(a1, *(_WORD *)(a2 + 32) & 0xFFFA);
        v10 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
        sub_9C8C60(a1, v10);
        result = sub_9C8C60(a1, *(unsigned __int16 *)(*(_QWORD *)(a2 + 112) + 32LL));
        break;
      case 1:
      case 4:
        if ( !(_WORD)v9 )
          v9 = *(_QWORD *)(a2 + 104);
        sub_D953B0(a1, v9, a3, a4, a5, a6);
        sub_9C8C60(a1, *(_WORD *)(a2 + 32) & 0xFFFA);
        v34 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
        result = sub_9C8C60(a1, v34);
        break;
      default:
        break;
    }
  }
  else if ( (int)result > 297 )
  {
    result = (unsigned int)(result - 298);
    switch ( (int)result )
    {
      case 0:
      case 1:
      case 40:
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 55:
      case 64:
      case 65:
      case 66:
      case 67:
        v9 = *(unsigned __int16 *)(a2 + 96);
        goto LABEL_7;
      case 25:
LABEL_47:
        result = sub_D953B0(a1, *(_QWORD *)(a2 + 96), a3, a4, a5, a6);
        break;
      case 68:
      case 69:
        if ( *(__int64 *)(a2 + 104) >= 0 )
        {
          sub_D953B0(a1, *(_QWORD *)(a2 + 96), a3, a4, a5, a6);
          result = sub_D953B0(a1, *(_QWORD *)(a2 + 104), v54, v55, v56, v57);
        }
        break;
      case 74:
LABEL_44:
        sub_D953B0(a1, *(_QWORD *)(a2 + 96), a3, a4, a5, a6);
        v47 = *(_QWORD *)(a2 + 104);
        goto LABEL_45;
      default:
        break;
    }
  }
  else if ( (int)result > 165 )
  {
    if ( (_DWORD)result == 235 )
    {
LABEL_31:
      sub_9C8C60(a1, *(_DWORD *)(a2 + 96));
      result = sub_9C8C60(a1, *(_DWORD *)(a2 + 100));
    }
  }
  else if ( (int)result > 4 )
  {
    result = (unsigned int)(result - 5);
    switch ( (int)result )
    {
      case 0:
        result = sub_D953B0(a1, 1LL << *(_BYTE *)(a2 + 96), a3, *(unsigned __int8 *)(a2 + 96), a5, a6);
        goto LABEL_10;
      case 1:
      case 5:
      case 7:
      case 31:
        goto LABEL_47;
      case 4:
      case 10:
      case 34:
        goto LABEL_49;
      case 6:
      case 30:
        sub_D953B0(a1, *(_QWORD *)(a2 + 96), a3, a4, a5, a6);
        result = sub_9C8C60(a1, (*(_BYTE *)(a2 + 32) & 8) != 0);
        goto LABEL_10;
      case 8:
      case 9:
      case 14:
      case 32:
      case 33:
      case 38:
        goto LABEL_44;
      case 11:
      case 35:
        goto LABEL_31;
      case 12:
      case 36:
        sub_D953B0(a1, 1LL << *(_BYTE *)(a2 + 108), a3, *(unsigned __int8 *)(a2 + 108), a5, a6);
        sub_9C8C60(a1, *(_DWORD *)(a2 + 104) & 0x7FFFFFFF);
        if ( *(int *)(a2 + 104) < 0 )
        {
          (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a2 + 96) + 40LL))(*(_QWORD *)(a2 + 96), a1);
          goto LABEL_46;
        }
        v47 = *(_QWORD *)(a2 + 96);
        break;
      case 13:
      case 37:
      case 39:
        BUG();
      case 40:
        sub_9C8C60(a1, *(_DWORD *)(a2 + 100));
        sub_D953B0(a1, *(_QWORD *)(a2 + 104), v48, v49, v50, v51);
LABEL_49:
        result = sub_9C8C60(a1, *(_DWORD *)(a2 + 96));
        goto LABEL_10;
      case 160:
        v58 = *(unsigned __int16 **)(a2 + 48);
        v59 = *v58;
        v60 = *((_QWORD *)v58 + 1);
        v64 = v59;
        v65 = v60;
        if ( v59 )
        {
          if ( (unsigned __int16)(v59 - 176) <= 0x34u )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
          }
          LODWORD(result) = word_4456340[v59 - 1];
        }
        else
        {
          if ( sub_3007100((__int64)&v64) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          LODWORD(result) = sub_3007130((__int64)&v64, a2);
        }
        v61 = *(int **)(a2 + 96);
        result = (unsigned int)result;
        v62 = &v61[(unsigned int)result];
        if ( v62 != v61 )
        {
          result = *(unsigned int *)(a1 + 8);
          do
          {
            v63 = *v61;
            if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
            {
              sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 4u, a5, a6);
              result = *(unsigned int *)(a1 + 8);
            }
            ++v61;
            *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = v63;
            result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
            *(_DWORD *)(a1 + 8) = result;
          }
          while ( v62 != v61 );
        }
        goto LABEL_10;
      default:
        goto LABEL_10;
    }
LABEL_45:
    sub_D953B0(a1, v47, v43, v44, v45, v46);
LABEL_46:
    result = sub_9C8C60(a1, *(_DWORD *)(a2 + 112));
  }
LABEL_10:
  if ( (*(_BYTE *)(a2 + 32) & 2) != 0 )
  {
    v11 = *(unsigned int *)(a1 + 8);
    v12 = *(_WORD *)(a2 + 32) & 0xFFFA;
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v11 + 1, 4u, a5, a6);
      v11 = *(unsigned int *)(a1 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v11) = v12;
    ++*(_DWORD *)(a1 + 8);
    v15 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
    v16 = *(unsigned int *)(a1 + 8);
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v16 + 1, 4u, v13, v14);
      v16 = *(unsigned int *)(a1 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v16) = v15;
    v17 = *(unsigned int *)(a1 + 12);
    v18 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v18;
    v19 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 112) + 32LL);
    if ( v18 + 1 > v17 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v18 + 1, 4u, v13, v14);
      v18 = *(unsigned int *)(a1 + 8);
    }
    v20 = -1;
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v18) = v19;
    v21 = -1;
    v22 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v22;
    v23 = *(_QWORD *)(a2 + 112);
    v24 = *(_QWORD *)(v23 + 24);
    if ( (v24 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
    {
      v35 = *(_BYTE *)(v23 + 24);
      v36 = v24 >> 3;
      v13 = v35 & 6;
      v37 = v35 & 2;
      if ( (_BYTE)v13 == 2 || (v35 & 1) != 0 )
      {
        v52 = HIDWORD(v24);
        v53 = HIWORD(v24);
        if ( !v37 )
          v53 = v52;
        v42 = (v53 + 7) >> 3;
      }
      else
      {
        v38 = v24;
        v39 = v24;
        v40 = HIDWORD(v24);
        v41 = v38 >> 8;
        v13 = HIWORD(v39);
        if ( v37 )
          LODWORD(v40) = v13;
        v42 = ((unsigned __int64)((unsigned __int16)v41 * (unsigned int)v40) + 7) >> 3;
        if ( (v36 & 1) != 0 )
          v42 |= 0x4000000000000000uLL;
      }
      v20 = v42;
      v21 = HIDWORD(v42);
    }
    if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v22 + 1, 4u, v13, v14);
      v22 = *(unsigned int *)(a1 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v22) = v20;
    v25 = *(unsigned int *)(a1 + 12);
    v26 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v26;
    if ( v26 + 1 > v25 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v26 + 1, 4u, v13, v14);
      v26 = *(unsigned int *)(a1 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v26) = v21;
    v27 = *(unsigned int *)(a1 + 12);
    v28 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v28;
    v29 = *(unsigned __int16 *)(a2 + 96);
    if ( !(_WORD)v29 )
      v29 = *(_QWORD *)(a2 + 104);
    v30 = v29;
    v31 = v29;
    if ( v28 + 1 > v27 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v28 + 1, 4u, v13, v14);
      v28 = *(unsigned int *)(a1 + 8);
    }
    v32 = HIDWORD(v30);
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v28) = v31;
    v33 = *(unsigned int *)(a1 + 12);
    result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = result;
    if ( result + 1 > v33 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 4u, v13, v14);
      result = *(unsigned int *)(a1 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = v32;
    ++*(_DWORD *)(a1 + 8);
  }
  return result;
}
