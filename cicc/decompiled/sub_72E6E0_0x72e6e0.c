// Function: sub_72E6E0
// Address: 0x72e6e0
//
__int64 __fastcall sub_72E6E0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // dl
  __int64 result; // rax
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // ebx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // rdi
  int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r12
  __int64 v23; // rsi
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 *v29; // rdi
  int v30; // ebx

  v2 = *(_BYTE *)(a1 + 24);
  result = (unsigned int)v2 + 31 * dword_4F07AD0;
  dword_4F07AD0 = v2 + 31 * dword_4F07AD0;
  switch ( v2 )
  {
    case 1u:
    case 0x17u:
      result = (_DWORD)result + 8 * *(unsigned __int8 *)(a1 + 56) - (unsigned int)*(unsigned __int8 *)(a1 + 56);
      dword_4F07AD0 = result;
      break;
    case 3u:
      v6 = *(_QWORD *)(a1 + 56);
      v9 = sub_72E220(v6, a2);
      result = *(_QWORD *)(v6 + 216);
      if ( result )
      {
        result = sub_72E120(*(__int64 **)result, a2, v7, v8, v10, v11);
        v9 += result;
      }
      dword_4F07AD0 += v9;
      break;
    case 4u:
      result = sub_72E220(*(_QWORD *)(a1 + 56), a2);
      dword_4F07AD0 += result;
      break;
    case 7u:
      v21 = *(_QWORD *)(a1 + 56);
      v22 = *(_QWORD *)(v21 + 16);
      v23 = ((*(_BYTE *)v21 & 2) != 0)
          + 2
          * (((*(_BYTE *)v21 & 8) != 0)
           + 2
           * (((*(_BYTE *)v21 & 0x10) != 0)
            + 2
            * (((*(_BYTE *)v21 & 0x20) != 0)
             + 2
             * (((*(_BYTE *)v21 & 0x40) != 0)
              + 2
              * ((*(_BYTE *)v21 >> 7)
               + 2 * ((*(_BYTE *)(v21 + 1) & 1) + 2 * (unsigned int)((*(_BYTE *)(v21 + 1) & 2) != 0)))))));
      v4 = result + 2 * ((*(_BYTE *)v21 & 1) + 2 * v23);
      dword_4F07AD0 = v4;
      if ( v22 )
      {
        LODWORD(result) = *(_DWORD *)(v22 + 168);
        if ( !(_DWORD)result )
        {
          v24 = sub_72E220(v22, v23);
          v29 = *(__int64 **)(v22 + 240);
          v30 = v24;
          if ( v29 )
            v30 = sub_72E120(v29, v23, v25, v26, v27, v28) + v24;
          LODWORD(result) = 1;
          v4 = dword_4F07AD0;
          if ( v30 )
            LODWORD(result) = v30;
          *(_DWORD *)(v22 + 168) = result;
        }
        goto LABEL_3;
      }
      break;
    case 0xBu:
      result = (unsigned int)result + 2 * (*(_BYTE *)(a1 + 64) & 1);
      dword_4F07AD0 = result;
      break;
    case 0xDu:
      if ( *(_BYTE *)(a1 + 57) )
      {
        v20 = sub_89A800(*(_QWORD *)(a1 + 64));
        result = sub_72E220(v20, a2);
        dword_4F07AD0 += result;
      }
      break;
    case 0xEu:
      result = (unsigned int)result + 2 * *(unsigned __int8 *)(a1 + 57);
      dword_4F07AD0 = result;
      break;
    case 0x14u:
      v12 = *(_QWORD *)(a1 + 56);
      v4 = *(_DWORD *)(v12 + 168);
      if ( !v4 )
      {
        v13 = sub_72E220(*(_QWORD *)(a1 + 56), a2);
        v18 = *(__int64 **)(v12 + 240);
        v19 = v13;
        if ( v18 )
          v19 = sub_72E120(v18, a2, v14, v15, v16, v17) + v13;
        v4 = 1;
        LODWORD(result) = dword_4F07AD0;
        if ( v19 )
          v4 = v19;
        *(_DWORD *)(v12 + 168) = v4;
      }
      goto LABEL_3;
    case 0x18u:
      v4 = *(_DWORD *)(a1 + 60) + 7 * *(_DWORD *)(a1 + 56);
LABEL_3:
      result = (unsigned int)(v4 + result);
      dword_4F07AD0 = result;
      break;
    case 0x1Eu:
      result = 15 * *(unsigned __int16 *)(a1 + 64)
             + (_DWORD)result
             + 8 * (*(_BYTE *)(a1 + 66) & 1)
             - (*(_BYTE *)(a1 + 66) & 1u);
      dword_4F07AD0 = result;
      break;
    case 0x20u:
    case 0x25u:
      v5 = sub_89A800(*(_QWORD *)(a1 + 56));
      result = sub_72E220(v5, a2);
      dword_4F07AD0 += result;
      break;
    default:
      return result;
  }
  return result;
}
