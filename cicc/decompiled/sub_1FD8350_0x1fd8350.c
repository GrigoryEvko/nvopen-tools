// Function: sub_1FD8350
// Address: 0x1fd8350
//
char __fastcall sub_1FD8350(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // rbp
  __int64 v5; // r12
  __int64 v6; // r13
  char result; // al
  __int64 **v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 *v14; // rbx
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  char v19; // r15
  _QWORD *v20; // rax
  unsigned int v21; // edx
  char v22; // r8
  unsigned int v23; // eax
  unsigned int v24; // r14d
  unsigned int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // edx
  __int64 v30; // rdi
  int v31; // ecx
  unsigned int v32; // edx
  __int64 v33; // rsi
  int v34; // eax
  __int64 (*v35)(); // rax
  __int64 v36; // rcx
  __int64 v37; // rbx
  unsigned int v38; // esi
  __int64 v39; // rdi
  unsigned int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // r9
  __int64 v43; // rsi
  unsigned int v44; // eax
  int v45; // r11d
  __int64 *v46; // r10
  int v47; // edi
  int v48; // edi
  unsigned int v49; // [rsp-5Ch] [rbp-5Ch]
  __int64 v50; // [rsp-58h] [rbp-58h] BYREF
  __int64 v51; // [rsp-50h] [rbp-50h]
  __int64 *v52; // [rsp-48h] [rbp-48h] BYREF
  __int64 v53; // [rsp-40h] [rbp-40h]
  __int64 v54; // [rsp-28h] [rbp-28h]
  __int64 v55; // [rsp-20h] [rbp-20h]
  __int64 v56; // [rsp-8h] [rbp-8h]

  v56 = v4;
  v55 = v6;
  v54 = v5;
  switch ( a3 )
  {
    case 0u:
    case 1u:
    case 3u:
    case 4u:
    case 5u:
    case 6u:
    case 8u:
    case 9u:
    case 0xAu:
    case 0x1Eu:
    case 0x1Fu:
    case 0x21u:
    case 0x22u:
    case 0x23u:
    case 0x27u:
    case 0x29u:
    case 0x2Bu:
    case 0x2Cu:
    case 0x30u:
    case 0x31u:
    case 0x32u:
    case 0x33u:
    case 0x34u:
    case 0x35u:
    case 0x37u:
    case 0x38u:
    case 0x39u:
    case 0x3Au:
    case 0x3Bu:
    case 0x3Cu:
    case 0x3Du:
      return 0;
    case 2u:
      result = 0;
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
        return result;
      v36 = *(_QWORD *)(a2 - 24);
      v37 = a1[5];
      v50 = v36;
      v38 = *(_DWORD *)(v37 + 72);
      if ( !v38 )
      {
        ++*(_QWORD *)(v37 + 48);
LABEL_88:
        v38 *= 2;
        goto LABEL_89;
      }
      v39 = *(_QWORD *)(v37 + 56);
      v40 = (v38 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v41 = (__int64 *)(v39 + 16LL * v40);
      v42 = *v41;
      if ( v36 == *v41 )
      {
LABEL_55:
        v43 = v41[1];
        goto LABEL_56;
      }
      v45 = 1;
      v46 = 0;
      while ( v42 != -8 )
      {
        if ( v42 == -16 && !v46 )
          v46 = v41;
        v40 = (v38 - 1) & (v45 + v40);
        v41 = (__int64 *)(v39 + 16LL * v40);
        v42 = *v41;
        if ( v36 == *v41 )
          goto LABEL_55;
        ++v45;
      }
      v47 = *(_DWORD *)(v37 + 64);
      if ( v46 )
        v41 = v46;
      ++*(_QWORD *)(v37 + 48);
      v48 = v47 + 1;
      if ( 4 * v48 >= 3 * v38 )
        goto LABEL_88;
      if ( v38 - *(_DWORD *)(v37 + 68) - v48 > v38 >> 3 )
        goto LABEL_84;
LABEL_89:
      sub_1D52F30(v37 + 48, v38);
      sub_1FD4470(v37 + 48, &v50, &v52);
      v41 = v52;
      v36 = v50;
LABEL_84:
      ++*(_DWORD *)(v37 + 64);
      if ( *v41 != -8 )
        --*(_DWORD *)(v37 + 68);
      *v41 = v36;
      v43 = 0;
      v41[1] = 0;
LABEL_56:
      sub_1FD3DF0((__int64)a1, v43, a2 + 48);
      return 1;
    case 7u:
      result = 1;
      if ( (*(_BYTE *)(a1[11] + 808LL) & 0x10) == 0 )
        return result;
      v35 = *(__int64 (**)())(*a1 + 56LL);
      if ( v35 == sub_1FD34B0 )
        return 0;
      return ((unsigned int (__fastcall *)(_QWORD *, __int64, __int64, __int64))v35)(a1, 1, 1, 215) != 0;
    case 0xBu:
      v27 = 52;
      return sub_1FDC220(a1, a2, v27);
    case 0xCu:
      v27 = 76;
      return sub_1FDC220(a1, a2, v27);
    case 0xDu:
      v27 = 53;
      return sub_1FDC220(a1, a2, v27);
    case 0xEu:
      if ( sub_15FB6D0(a2, 0, a3, a4) )
        return sub_1FDCBD0(a1, a2);
      v27 = 77;
      return sub_1FDC220(a1, a2, v27);
    case 0xFu:
      v27 = 54;
      return sub_1FDC220(a1, a2, v27);
    case 0x10u:
      v27 = 78;
      return sub_1FDC220(a1, a2, v27);
    case 0x11u:
      v27 = 56;
      return sub_1FDC220(a1, a2, v27);
    case 0x12u:
      v27 = 55;
      return sub_1FDC220(a1, a2, v27);
    case 0x13u:
      v27 = 79;
      return sub_1FDC220(a1, a2, v27);
    case 0x14u:
      v27 = 58;
      return sub_1FDC220(a1, a2, v27);
    case 0x15u:
      v27 = 57;
      return sub_1FDC220(a1, a2, v27);
    case 0x16u:
      v27 = 80;
      return sub_1FDC220(a1, a2, v27);
    case 0x17u:
      v27 = 122;
      return sub_1FDC220(a1, a2, v27);
    case 0x18u:
      v27 = 124;
      return sub_1FDC220(a1, a2, v27);
    case 0x19u:
      v27 = 123;
      return sub_1FDC220(a1, a2, v27);
    case 0x1Au:
      v27 = 118;
      return sub_1FDC220(a1, a2, v27);
    case 0x1Bu:
      v27 = 119;
      return sub_1FDC220(a1, a2, v27);
    case 0x1Cu:
      v27 = 120;
      return sub_1FDC220(a1, a2, v27);
    case 0x1Du:
      v28 = a1[5];
      result = 0;
      v29 = *(_DWORD *)(v28 + 360);
      if ( !v29 )
        return result;
      v30 = *(_QWORD *)(v28 + 344);
      v31 = v29 - 1;
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v33 = *(_QWORD *)(v30 + 16LL * v32);
      result = 1;
      if ( a2 == v33 )
        return result;
      v34 = 1;
      while ( 2 )
      {
        if ( v33 == -8 )
          return 0;
        v32 = v31 & (v34 + v32);
        v33 = *(_QWORD *)(v30 + 16LL * v32);
        if ( a2 != v33 )
        {
          ++v34;
          continue;
        }
        break;
      }
      return 1;
    case 0x20u:
      return sub_1FDC620();
    case 0x24u:
      goto LABEL_24;
    case 0x25u:
      goto LABEL_26;
    case 0x26u:
      v26 = 142;
      return sub_1FDBB70(a1, a2, v26);
    case 0x28u:
      v26 = 152;
      return sub_1FDBB70(a1, a2, v26);
    case 0x2Au:
      v26 = 146;
      return sub_1FDBB70(a1, a2, v26);
    case 0x2Du:
    case 0x2Eu:
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v10 = *(__int64 ***)(a2 - 8);
      else
        v10 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      LOBYTE(v11) = sub_1FD35E0(a1[12], **v10);
      v13 = v12;
      v14 = v11;
      LOBYTE(v15) = sub_1FD35E0(a1[12], *(_QWORD *)a2);
      v52 = v14;
      v18 = (unsigned int)v14;
      LODWORD(v50) = v15;
      v19 = v15;
      v51 = v16;
      v53 = v13;
      if ( (_BYTE)v14 == (_BYTE)v15 )
      {
        if ( (_BYTE)v14 )
          goto LABEL_8;
        if ( v13 == v16 )
        {
          LOBYTE(v14) = 0;
          goto LABEL_19;
        }
      }
      else if ( (_BYTE)v15 )
      {
        v49 = sub_1FD3510(v15);
LABEL_14:
        if ( v22 )
          v23 = sub_1FD3510(v22);
        else
          v23 = sub_1F58D40((__int64)&v52);
        if ( v23 < v49 )
        {
LABEL_26:
          v26 = 143;
          return sub_1FDBB70(a1, a2, v26);
        }
        v52 = v14;
        v53 = v13;
        if ( v19 != (_BYTE)v14 )
        {
          if ( v19 )
          {
            v24 = sub_1FD3510(v19);
LABEL_21:
            if ( (_BYTE)v14 )
              v25 = sub_1FD3510((char)v14);
            else
              v25 = sub_1F58D40((__int64)&v52);
            if ( v25 > v24 )
            {
LABEL_24:
              v26 = 145;
              return sub_1FDBB70(a1, a2, v26);
            }
LABEL_8:
            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
            {
              v20 = *(_QWORD **)(a2 - 8);
            }
            else
            {
              v16 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
              v20 = (_QWORD *)(a2 - v16);
            }
            v21 = sub_1FD8F60(a1, *v20, v16, v17, v18);
            result = 0;
            if ( v21 )
            {
              sub_1FD5CC0((__int64)a1, a2, v21, 1);
              return 1;
            }
            return result;
          }
LABEL_20:
          v24 = sub_1F58D40((__int64)&v50);
          goto LABEL_21;
        }
        if ( (_BYTE)v14 )
          goto LABEL_8;
LABEL_19:
        if ( v13 == v51 )
          goto LABEL_8;
        goto LABEL_20;
      }
      v44 = sub_1F58D40((__int64)&v50);
      v22 = (char)v14;
      v49 = v44;
      goto LABEL_14;
    case 0x2Fu:
      return sub_1FDBD10();
    case 0x36u:
      return sub_1FDB9A0();
    case 0x3Eu:
      return sub_1FD8330((__int64)a1, a2);
    default:
      return 0;
  }
}
