// Function: sub_174BF40
// Address: 0x174bf40
//
__int64 ***__fastcall sub_174BF40(__int64 *a1, __int64 a2, __int64 **a3, unsigned __int8 a4)
{
  _QWORD *v5; // r13
  int v6; // eax
  __int64 v7; // rbx
  __int64 ***v8; // r12
  int v10; // ebx
  _QWORD *v11; // rax
  __int64 *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  char v16; // dl
  unsigned int v17; // ebx
  __int64 v18; // rax
  _QWORD *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rax
  int v24; // eax
  __int64 v25; // rax
  unsigned __int8 v26; // r11
  __int64 **v27; // rbx
  __int64 v29; // r13
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rcx
  __int64 v36; // r10
  __int64 v37; // rdx
  __int64 v38; // rdx
  int v39; // ecx
  __int64 v40; // rcx
  __int64 *v41; // rdx
  __int64 v42; // rsi
  unsigned __int64 v43; // rcx
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // r15
  __int64 v49; // rsi
  unsigned __int8 *v50; // rsi
  __int64 v51; // rcx
  __int64 ****v52; // rax
  __int64 v53; // [rsp+8h] [rbp-78h]
  __int64 v54; // [rsp+10h] [rbp-70h]
  __int64 v55; // [rsp+18h] [rbp-68h]
  unsigned int v57; // [rsp+24h] [rbp-5Ch]
  unsigned int v58; // [rsp+28h] [rbp-58h]
  int v59; // [rsp+28h] [rbp-58h]
  __int64 v60; // [rsp+28h] [rbp-58h]
  __int64 v61[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v62; // [rsp+40h] [rbp-40h]

  v5 = (_QWORD *)a2;
  v6 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v6 > 0x10u )
  {
    v10 = v6 - 24;
    switch ( *(_BYTE *)(a2 + 16) )
    {
      case '#':
      case '%':
      case '\'':
      case ')':
      case ',':
      case '/':
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v11 = *(_QWORD **)(a2 - 8);
        else
          v11 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v58 = a4;
        v12 = (__int64 *)sub_174BF40(a1, *v11, a3, a4);
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v13 = *(_QWORD *)(a2 - 8);
        else
          v13 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v14 = sub_174BF40(a1, *(_QWORD *)(v13 + 24), a3, v58);
        v62 = 257;
        v8 = (__int64 ***)sub_15FB440(v10, v12, v14, (__int64)v61, 0);
        v15 = (__int64)v8;
        goto LABEL_45;
      case '$':
      case '&':
      case '(':
      case '*':
      case '+':
      case '-':
      case '.':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case ':':
      case ';':
      case '?':
      case '@':
      case 'A':
      case 'B':
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
      case 'N':
      case 'O':
        v17 = a4;
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v18 = *(_QWORD *)(a2 - 8);
        else
          v18 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v19 = (_QWORD *)sub_174BF40(a1, *(_QWORD *)(v18 + 24), a3, a4);
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v20 = *(_QWORD *)(a2 - 8);
        else
          v20 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v21 = sub_174BF40(a1, *(_QWORD *)(v20 + 48), a3, v17);
        v62 = 257;
        v22 = v21;
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v23 = *(__int64 **)(a2 - 8);
        else
          v23 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v8 = (__int64 ***)sub_14EDD70(*v23, v19, v22, (__int64)v61, 0, 0);
        v15 = (__int64)v8;
        goto LABEL_45;
      case '<':
      case '=':
      case '>':
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        {
          v52 = *(__int64 *****)(a2 - 8);
          v8 = *v52;
          if ( **v52 == a3 )
            return v8;
          v62 = 257;
          v8 = *v52;
          v16 = v10 == 38;
        }
        else
        {
          v8 = *(__int64 ****)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
          if ( a3 == *v8 )
            return v8;
          v62 = 257;
          v16 = v10 == 38;
        }
        v8 = (__int64 ***)sub_15FE0A0(v8, (__int64)a3, v16, (__int64)v61, 0);
        v15 = (__int64)v8;
LABEL_45:
        sub_164B7C0(v15, (__int64)v5);
        v47 = v5[6];
        v61[0] = v47;
        if ( v47 )
        {
          v48 = (__int64)(v8 + 6);
          sub_1623A60((__int64)v61, v47, 2);
          v49 = (__int64)v8[6];
          if ( !v49 )
            goto LABEL_48;
        }
        else
        {
          v49 = (__int64)v8[6];
          v48 = (__int64)(v8 + 6);
          if ( !v49 )
          {
LABEL_50:
            sub_157E9D0(v5[5] + 40LL, (__int64)v8);
            v51 = v5[3];
            v8[4] = (__int64 **)(v5 + 3);
            v51 &= 0xFFFFFFFFFFFFFFF8LL;
            v8[3] = (__int64 **)(v51 | (unsigned __int64)v8[3] & 7);
            *(_QWORD *)(v51 + 8) = v8 + 3;
            v5[3] = v5[3] & 7LL | (unsigned __int64)(v8 + 3);
            sub_170B990(*a1, (__int64)v8);
            return v8;
          }
        }
        sub_161E7C0(v48, v49);
LABEL_48:
        v50 = (unsigned __int8 *)v61[0];
        v8[6] = (__int64 **)v61[0];
        if ( v50 )
          sub_1623210((__int64)v61, v50, v48);
        goto LABEL_50;
      case 'M':
        v24 = *(_DWORD *)(a2 + 20);
        v62 = 257;
        v59 = v24 & 0xFFFFFFF;
        v25 = sub_1648B60(64);
        v26 = a4;
        v8 = (__int64 ***)v25;
        if ( v25 )
        {
          v15 = v25;
          sub_15F1EA0(v25, (__int64)a3, 53, 0, 0, 0);
          *((_DWORD *)v8 + 14) = v59;
          sub_164B780((__int64)v8, v61);
          sub_1648880((__int64)v8, *((_DWORD *)v8 + 14), 1);
          v26 = a4;
        }
        else
        {
          v15 = 0;
        }
        if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
        {
          v60 = v15;
          v57 = v26;
          v27 = a3;
          v55 = 8LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v29 = 0;
          do
          {
            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
              v30 = *(_QWORD *)(a2 - 8);
            else
              v30 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
            v31 = *(_QWORD *)(v30 + 3 * v29);
            v32 = sub_174BF40(a1, v31, v27, v57);
            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
              v35 = *(_QWORD *)(a2 - 8);
            else
              v35 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
            v36 = *(_QWORD *)(v29 + v35 + 24LL * *(unsigned int *)(a2 + 56) + 8);
            v37 = *((_DWORD *)v8 + 5) & 0xFFFFFFF;
            if ( (_DWORD)v37 == *((_DWORD *)v8 + 14) )
            {
              v53 = *(_QWORD *)(v29 + v35 + 24LL * *(unsigned int *)(a2 + 56) + 8);
              v54 = v32;
              sub_15F55D0((__int64)v8, v31, v37, v35, v33, v34);
              v36 = v53;
              v32 = v54;
              LODWORD(v37) = *((_DWORD *)v8 + 5) & 0xFFFFFFF;
            }
            v38 = ((_DWORD)v37 + 1) & 0xFFFFFFF;
            v39 = v38 | *((_DWORD *)v8 + 5) & 0xF0000000;
            *((_DWORD *)v8 + 5) = v39;
            if ( (v39 & 0x40000000) != 0 )
              v40 = (__int64)*(v8 - 1);
            else
              v40 = v60 - 24 * v38;
            v41 = (__int64 *)(v40 + 24LL * (unsigned int)(v38 - 1));
            if ( *v41 )
            {
              v42 = v41[1];
              v43 = v41[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v43 = v42;
              if ( v42 )
                *(_QWORD *)(v42 + 16) = *(_QWORD *)(v42 + 16) & 3LL | v43;
            }
            *v41 = v32;
            if ( v32 )
            {
              v44 = *(_QWORD *)(v32 + 8);
              v41[1] = v44;
              if ( v44 )
                *(_QWORD *)(v44 + 16) = (unsigned __int64)(v41 + 1) | *(_QWORD *)(v44 + 16) & 3LL;
              v41[2] = v41[2] & 3 | (v32 + 8);
              *(_QWORD *)(v32 + 8) = v41;
            }
            v45 = *((_DWORD *)v8 + 5) & 0xFFFFFFF;
            if ( (*((_BYTE *)v8 + 23) & 0x40) != 0 )
              v46 = (__int64)*(v8 - 1);
            else
              v46 = v60 - 24 * v45;
            v29 += 8;
            *(_QWORD *)(v46 + 8LL * (unsigned int)(v45 - 1) + 24LL * *((unsigned int *)v8 + 14) + 8) = v36;
          }
          while ( v55 != v29 );
          v15 = v60;
          v5 = (_QWORD *)a2;
        }
        goto LABEL_45;
    }
  }
  v7 = sub_15A4750((__int64 ***)a2, a3, a4);
  v8 = (__int64 ***)sub_14DBA30(v7, a1[333], a1[331]);
  if ( !v8 )
    return (__int64 ***)v7;
  return v8;
}
