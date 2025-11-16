// Function: sub_27B2E30
// Address: 0x27b2e30
//
__int64 __fastcall sub_27B2E30(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 v6; // rdx
  unsigned __int8 *v7; // r8
  unsigned int v8; // r13d
  int v10; // edx
  int v11; // edx
  __int64 v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rcx
  int v15; // ebx
  __int64 v16; // r14
  unsigned int v17; // esi
  __int64 v18; // rcx
  int v19; // ebx
  __int64 v20; // r8
  unsigned int v21; // edx
  __int64 v22; // rax
  __int64 v23; // r11
  int v24; // r10d
  __int64 *v25; // rbx
  __int64 v26; // rsi
  unsigned int v27; // esi
  int v28; // eax
  __int64 v29; // rdx
  int v30; // eax
  int v31; // eax
  int v32; // edx
  unsigned __int64 v33; // rcx
  __int64 v34; // rbx
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 *i; // [rsp+10h] [rbp-60h]
  __int64 v37[2]; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v38; // [rsp+28h] [rbp-48h] BYREF
  unsigned __int64 v39; // [rsp+30h] [rbp-40h] BYREF
  __int64 v40[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a1 + 24);
  v37[0] = (__int64)a2;
  v4 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)v3 )
    goto LABEL_8;
  v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = v4 + 16LL * v5;
  v7 = *(unsigned __int8 **)v6;
  if ( a2 != *(unsigned __int8 **)v6 )
  {
    v10 = 1;
    while ( v7 != (unsigned __int8 *)-4096LL )
    {
      v24 = v10 + 1;
      v5 = (v3 - 1) & (v10 + v5);
      v6 = v4 + 16LL * v5;
      v7 = *(unsigned __int8 **)v6;
      if ( a2 == *(unsigned __int8 **)v6 )
        goto LABEL_3;
      v10 = v24;
    }
LABEL_8:
    v11 = *a2;
    if ( (unsigned __int8)v11 <= 0x1Cu )
    {
LABEL_16:
      v15 = *(_DWORD *)(a1 + 272);
      *(_DWORD *)sub_10E84F0(a1, v37) = v15;
      v8 = *(_DWORD *)(a1 + 272);
      *(_DWORD *)(a1 + 272) = v8 + 1;
      return v8;
    }
    v12 = *((_QWORD *)a2 + 5);
    if ( *(_BYTE *)(a1 + 308) )
    {
      v13 = *(_QWORD **)(a1 + 288);
      v14 = &v13[*(unsigned int *)(a1 + 300)];
      if ( v13 == v14 )
        return (unsigned int)-1;
      while ( v12 != *v13 )
      {
        if ( v14 == ++v13 )
          return (unsigned int)-1;
      }
    }
    else
    {
      if ( !sub_C8CA60(a1 + 280, v12) )
        return (unsigned int)-1;
      v11 = *a2;
    }
    switch ( v11 )
    {
      case '"':
      case ')':
      case '*':
      case '+':
      case ',':
      case '-':
      case '.':
      case '/':
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case ':':
      case ';':
      case '?':
      case 'C':
      case 'D':
      case 'E':
      case 'F':
      case 'G':
      case 'H':
      case 'I':
      case 'J':
      case 'K':
      case 'L':
      case 'M':
      case 'N':
      case 'O':
      case 'R':
      case 'S':
      case 'U':
      case 'V':
      case 'Z':
      case '[':
      case '\\':
      case '^':
        v16 = sub_27B4E80(a1, a2);
        if ( !v16 )
          goto LABEL_16;
        break;
      case '=':
      case '>':
        if ( byte_3F70480[8 * ((*((_WORD *)a2 + 1) >> 7) & 7) + 1] || sub_B46500(a2) )
          goto LABEL_16;
        v16 = sub_27B4E80(a1, a2);
        *(_BYTE *)(v16 + 52) = *((_WORD *)a2 + 1);
        *(_BYTE *)(v16 + 52) &= 1u;
        break;
      default:
        goto LABEL_16;
    }
    v17 = *(_DWORD *)(a1 + 56);
    v39 = v16;
    v35 = a1 + 32;
    if ( v17 )
    {
      v18 = *(_QWORD *)(a1 + 40);
      v19 = 1;
      v20 = 0;
      v21 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v22 = v18 + 16LL * v21;
      v23 = *(_QWORD *)v22;
      if ( *(_QWORD *)v22 == v16 )
      {
LABEL_21:
        v8 = *(_DWORD *)(v22 + 8);
        if ( v8 )
        {
LABEL_22:
          *(_DWORD *)sub_10E84F0(a1, v37) = v8;
          return v8;
        }
LABEL_29:
        v40[0] = *(_QWORD *)(v16 + 40);
        LODWORD(v38) = *(_DWORD *)(v16 + 12);
        v39 = sub_27B2C30((int *)&v38, v40, (_DWORD *)(v16 + 48), (_BYTE *)(v16 + 52), v16 + 56);
        v25 = *(__int64 **)(v16 + 24);
        for ( i = &v25[*(unsigned int *)(v16 + 36)]; i != v25; v39 = sub_103ECC0((__int64 *)&v39, v40) )
        {
          v26 = *v25++;
          LODWORD(v40[0]) = sub_27B2E30(a1, v26);
        }
        v38 = v39;
        if ( (unsigned __int8)sub_27B2460(a1 + 64, (__int64 *)&v38, &v39) )
        {
          v8 = *(_DWORD *)(v39 + 8);
          goto LABEL_22;
        }
        v27 = *(_DWORD *)(a1 + 88);
        v28 = *(_DWORD *)(a1 + 80);
        v29 = v39;
        ++*(_QWORD *)(a1 + 64);
        v30 = v28 + 1;
        v40[0] = v29;
        if ( 4 * v30 >= 3 * v27 )
        {
          v27 *= 2;
        }
        else if ( v27 - *(_DWORD *)(a1 + 84) - v30 > v27 >> 3 )
        {
LABEL_35:
          *(_DWORD *)(a1 + 80) = v30;
          if ( *(_QWORD *)v29 != -1 )
            --*(_DWORD *)(a1 + 84);
          *(_QWORD *)v29 = v38;
          v8 = *(_DWORD *)(a1 + 272);
          *(_DWORD *)(v29 + 8) = v8;
          *(_DWORD *)(a1 + 272) = v8 + 1;
          v40[0] = v16;
          *(_DWORD *)sub_27B2AF0(v35, v40) = v8;
          goto LABEL_22;
        }
        sub_9E25D0(a1 + 64, v27);
        sub_27B2460(a1 + 64, (__int64 *)&v38, v40);
        v29 = v40[0];
        v30 = *(_DWORD *)(a1 + 80) + 1;
        goto LABEL_35;
      }
      while ( v23 != -4096 )
      {
        if ( !v20 && v23 == -8192 )
          v20 = v22;
        v21 = (v17 - 1) & (v19 + v21);
        v22 = v18 + 16LL * v21;
        v23 = *(_QWORD *)v22;
        if ( *(_QWORD *)v22 == v16 )
          goto LABEL_21;
        ++v19;
      }
      if ( !v20 )
        v20 = v22;
      v31 = *(_DWORD *)(a1 + 48);
      ++*(_QWORD *)(a1 + 32);
      v32 = v31 + 1;
      v40[0] = v20;
      if ( 4 * (v31 + 1) < 3 * v17 )
      {
        v33 = v16;
        if ( v17 - *(_DWORD *)(a1 + 52) - v32 > v17 >> 3 )
        {
LABEL_48:
          *(_DWORD *)(a1 + 48) = v32;
          if ( *(_QWORD *)v20 != -4096 )
            --*(_DWORD *)(a1 + 52);
          *(_QWORD *)v20 = v33;
          *(_DWORD *)(v20 + 8) = 0;
          goto LABEL_29;
        }
        v34 = a1 + 32;
        sub_27B2910(v35, v17);
LABEL_53:
        sub_27B23A0(v34, (__int64 *)&v39, v40);
        v33 = v39;
        v20 = v40[0];
        v32 = *(_DWORD *)(a1 + 48) + 1;
        goto LABEL_48;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 32);
      v40[0] = 0;
    }
    v34 = a1 + 32;
    sub_27B2910(v35, 2 * v17);
    goto LABEL_53;
  }
LABEL_3:
  if ( v6 == v4 + 16 * v3 )
    goto LABEL_8;
  return *(unsigned int *)(v6 + 8);
}
