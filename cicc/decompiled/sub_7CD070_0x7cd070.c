// Function: sub_7CD070
// Address: 0x7cd070
//
unsigned __int8 **__fastcall sub_7CD070(__int64 a1, int a2, __int64 *a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 *v9; // rcx
  int v10; // r15d
  unsigned __int8 *v11; // r12
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // r13
  unsigned __int8 *v15; // rcx
  __int64 v16; // r13
  unsigned __int8 *v17; // rcx
  unsigned __int8 **result; // rax
  unsigned int v19; // edx
  int v20; // eax
  __int64 v21; // r15
  __int64 v22; // rdx
  int v23; // edx
  int v24; // r8d
  unsigned __int64 i; // r13
  unsigned __int64 *v26; // rcx
  unsigned int v27; // edi
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  signed __int8 v30; // r13
  int v31; // eax
  unsigned __int64 v32; // rsi
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // rax
  int v36; // eax
  int v37; // eax
  unsigned __int8 *v38; // r12
  int v39; // edi
  int v40; // eax
  int v41; // eax
  int v42; // edx
  int v43[2]; // [rsp+8h] [rbp-68h]
  int v44[2]; // [rsp+8h] [rbp-68h]
  int v45; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v48; // [rsp+10h] [rbp-60h]
  __int64 v49; // [rsp+10h] [rbp-60h]
  int v50[2]; // [rsp+18h] [rbp-58h]
  int v51[2]; // [rsp+18h] [rbp-58h]
  int v53; // [rsp+18h] [rbp-58h]
  int v54[2]; // [rsp+18h] [rbp-58h]
  int v55[2]; // [rsp+18h] [rbp-58h]
  int v56[2]; // [rsp+18h] [rbp-58h]
  int v57; // [rsp+2Ch] [rbp-44h] BYREF
  unsigned __int8 *v58; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int16 v59[28]; // [rsp+38h] [rbp-38h] BYREF

  v9 = *(unsigned __int64 **)a1;
  v10 = *(_DWORD *)(a1 + 16);
  v11 = **(unsigned __int8 ***)a1;
  v58 = v11;
  if ( v10 )
  {
    v15 = *(unsigned __int8 **)(a1 + 24);
    if ( v15 )
    {
      v16 = *v15;
      v17 = v15 + 1;
      if ( v10 == 1 )
        v17 = 0;
      *(_QWORD *)(a1 + 24) = v17;
    }
    else if ( *(_BYTE *)(a1 + 41) )
    {
      v16 = *(_QWORD *)(a1 + 32);
    }
    else
    {
      v16 = *v11++;
    }
    v13 = a4 & v16;
    *(_DWORD *)(a1 + 16) = v10 - 1;
  }
  else
  {
    v12 = *(_QWORD *)(a1 + 8);
    v13 = *v11;
    if ( !v12 || v11 != *(unsigned __int8 **)(v12 + 8) )
    {
      if ( *v11 )
      {
        if ( v13 == 92 && a2 )
        {
          v58 = v11 + 2;
          v14 = v11[1];
          switch ( (char)v14 )
          {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
              v23 = v11[2];
              v24 = 1;
              i = (int)v14 - 48;
              if ( (unsigned __int8)(v23 - 48) <= 7u )
              {
                v58 = v11 + 3;
                i = (v23 - 48) | (8 * i);
                v42 = v11[3];
                if ( (unsigned __int8)(v42 - 48) <= 7u )
                {
                  v58 = v11 + 4;
                  i = (v42 - 48) | (8 * i);
                }
              }
              goto LABEL_36;
            case 'E':
            case 'e':
              if ( !HIDWORD(qword_4F077B4) )
                goto LABEL_67;
              v11 += 2;
              v13 = a4 & 0x1B;
              goto LABEL_17;
            case 'U':
            case 'u':
              if ( !unk_4D042A0 )
                goto LABEL_32;
              *(_QWORD *)v43 = a4;
              v58 = v11;
              v28 = sub_7B39D0((unsigned __int64 *)&v58, 0, 0, 1);
              v24 = a5;
              a4 = *(_QWORD *)v43;
              i = v28;
              if ( !a5 )
                goto LABEL_36;
              v29 = sub_7CB990(a1, v28, a6);
              v11 = v58;
              v13 = *(_QWORD *)v43 & v29;
              goto LABEL_17;
            case 'a':
              if ( dword_4F077C4 == 1 )
                goto LABEL_67;
              v11 += 2;
              v13 = a4 & 7;
              goto LABEL_17;
            case 'b':
              v11 += 2;
              v13 = a4 & 8;
              goto LABEL_17;
            case 'f':
              v11 += 2;
              v13 = a4 & 0xC;
              goto LABEL_17;
            case 'n':
              v11 += 2;
              v13 = a4 & 0xA;
              goto LABEL_17;
            case 'r':
              v11 += 2;
              v13 = a4 & 0xD;
              goto LABEL_17;
            case 't':
              v11 += 2;
              v13 = a4 & 9;
              goto LABEL_17;
            case 'v':
              v11 += 2;
              v13 = a4 & 0xB;
              goto LABEL_17;
            case 'x':
              *(_QWORD *)v44 = a4;
              v48 = v9;
              v53 = v11[2];
              v30 = v11[2];
              v31 = isxdigit(v53);
              v26 = v48;
              a4 = *(_QWORD *)v44;
              if ( !v31 )
              {
                sub_7B0EB0(*v48 + 2, (__int64)dword_4F07508);
                v27 = 22;
                v13 = v44[0] & 0x78;
                if ( dword_4F077C4 != 1 && !unk_4D0436C )
                {
                  sub_6851C0(0x16u, dword_4F07508);
                  v11 = v58;
                  goto LABEL_17;
                }
LABEL_44:
                sub_684B30(v27, dword_4F07508);
                v11 = v58;
                goto LABEL_17;
              }
              v36 = 48;
              if ( (unsigned int)(v53 - 48) > 9 )
              {
                v37 = islower(v53);
                a4 = *(_QWORD *)v44;
                v26 = v48;
                v36 = v37 == 0 ? 55 : 87;
              }
              v38 = v11 + 3;
              for ( i = v30 - v36; ; i = (v45 - v39) | (16 * i) )
              {
                v58 = v38;
                v49 = a4;
                *(_QWORD *)v56 = v26;
                v45 = *v38;
                v41 = isxdigit(v45);
                v26 = *(unsigned __int64 **)v56;
                a4 = v49;
                if ( !v41 )
                  break;
                v39 = 48;
                if ( i > 0x7FFFFFFFFFFFFFFLL )
                  v10 = 1;
                if ( (unsigned int)(v45 - 48) > 9 )
                {
                  v40 = islower(v45);
                  a4 = v49;
                  v26 = *(unsigned __int64 **)v56;
                  v39 = v40 == 0 ? 55 : 87;
                }
                ++v38;
              }
              if ( v10 )
                goto LABEL_40;
              v24 = 1;
LABEL_36:
              if ( (i & ~a4) != 0 )
              {
                if ( !*(_BYTE *)(a1 + 41) || v24 )
                {
                  v26 = *(unsigned __int64 **)a1;
                  LOBYTE(v10) = v24;
LABEL_40:
                  *(_QWORD *)v51 = a4;
                  sub_7B0EB0(*v26, (__int64)dword_4F07508);
                  v13 = *(_QWORD *)v51 & i;
                  if ( dword_4F077C4 != 2 && dword_4D04964 && (v10 & 1) != 0 )
                  {
                    sub_684AC0(byte_4F07472[0], 0x1Bu);
                    v11 = v58;
                    goto LABEL_17;
                  }
                  v27 = 27;
                  goto LABEL_44;
                }
                *(_QWORD *)v55 = a4;
                v34 = sub_722B20(i, v59);
                a4 = *(_QWORD *)v55;
                if ( v34 == 2 )
                {
                  v35 = v59[1];
                  *(_QWORD *)(a1 + 24) = 0;
                  *(_DWORD *)(a1 + 16) = 1;
                  i = v59[0];
                  *(_QWORD *)(a1 + 32) = v35;
                }
              }
              v11 = v58;
              v13 = a4 & i;
              goto LABEL_17;
            default:
LABEL_32:
              if ( (unsigned __int8)(v14 - 34) <= 0x3Au
                && (v22 = 0x400000020000021LL, _bittest64(&v22, (unsigned int)(v14 - 34))) )
              {
                v13 = a4 & v14;
                v11 += 2;
              }
              else
              {
LABEL_67:
                *(_QWORD *)v54 = a4;
                sub_7B0EB0(*v9, (__int64)dword_4F07508);
                sub_684B30(0xC0u, dword_4F07508);
                v11 = v58;
                v13 = *(_QWORD *)v54 & v14;
              }
              goto LABEL_17;
          }
        }
      }
      else if ( !*(_BYTE *)(a1 + 44) )
      {
        v11 += 2;
        goto LABEL_17;
      }
      if ( dword_4D0432C )
      {
        *(_QWORD *)v50 = a4;
        v20 = sub_722680(v11, v59, &v57, unk_4F064A8 == 0);
        v21 = v20;
        a4 = *(_QWORD *)v50;
        if ( v57 )
        {
          v13 = 63;
          sub_7B0EB0((unsigned __int64)v58, (__int64)dword_4F07508);
          sub_684AC0(*(_BYTE *)(a1 + 42) == 0 ? 7 : 5, 0x366u);
          a4 = *(_QWORD *)v50;
          v11 = &v58[v21 - 1];
          goto LABEL_11;
        }
        v11 = v58;
        if ( !a6 )
        {
          *(_DWORD *)(a1 + 16) = v20 - 1;
          goto LABEL_11;
        }
        v32 = *(_QWORD *)v59;
        v58 = &v58[v20 - 1];
      }
      else
      {
        if ( !a6 )
        {
LABEL_11:
          ++v11;
          v13 &= a4;
          goto LABEL_17;
        }
        *(_QWORD *)v50 = a4;
        v32 = *v11;
      }
      v33 = sub_7CB990(a1, v32, 1);
      v11 = v58;
      a4 = *(_QWORD *)v50;
      v13 = v33;
      goto LABEL_11;
    }
    *(_QWORD *)(a1 + 8) = *(_QWORD *)v12;
    v19 = *(_DWORD *)(v12 + 16);
    *(_BYTE *)(v12 + 20) = 1;
    if ( v19 == 2 )
    {
      v11 += 2;
      v13 = a4 & 0xA;
    }
    else if ( v19 > 2 )
    {
      if ( v19 != 3 )
        sub_721090();
      v11 += 2;
      v13 = 0;
    }
    else
    {
      if ( v19 )
      {
        *(_DWORD *)(a1 + 16) = 1;
        *(_BYTE *)(a1 + 45) = 10;
        v13 = a4 & 0x5C;
      }
      else
      {
        *(_DWORD *)(a1 + 16) = 2;
        ++v11;
        *(_BYTE *)(a1 + 45) = 63;
        v13 = a4 & 0x3F;
        *(_BYTE *)(a1 + 46) = *(_BYTE *)(v12 + 24);
      }
      *(_QWORD *)(a1 + 24) = a1 + 45;
    }
  }
LABEL_17:
  *a3 = v13;
  result = *(unsigned __int8 ***)a1;
  **(_QWORD **)a1 = v11;
  return result;
}
