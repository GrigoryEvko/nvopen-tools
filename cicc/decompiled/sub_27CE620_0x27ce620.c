// Function: sub_27CE620
// Address: 0x27ce620
//
_QWORD *__fastcall sub_27CE620(_QWORD *a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  int v7; // eax
  unsigned __int8 *v9; // rbx
  __int64 v10; // rax
  unsigned __int8 *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int8 *v14; // r13
  _QWORD *v15; // rsi
  unsigned __int8 *v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // r14
  _QWORD *v19; // r13
  unsigned __int8 *v20; // rbx
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 v23; // rax

  v6 = *a2;
  if ( (_BYTE)v6 != 22 )
  {
    if ( (unsigned __int8)v6 <= 0x1Cu )
      v7 = *((unsigned __int16 *)a2 + 1);
    else
      v7 = v6 - 29;
    switch ( v7 )
    {
      case ' ':
      case '"':
      case '1':
      case '2':
        if ( (a2[7] & 0x40) != 0 )
          v9 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v9 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v10 = *(_QWORD *)v9;
        goto LABEL_11;
      case '0':
        v19 = a1 + 2;
        if ( !(_BYTE)qword_4FFD608 || (unsigned int)sub_DF9B70(a3) == -1 )
        {
          if ( (a2[7] & 0x40) != 0 )
            v20 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v20 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v21 = *(_QWORD *)v20;
          if ( (*(_BYTE *)(*(_QWORD *)v20 + 7LL) & 0x40) != 0 )
            v22 = *(__int64 **)(v21 - 8);
          else
            v22 = (__int64 *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF));
          v23 = *v22;
          *a1 = v19;
          a1[2] = v23;
          a1[1] = 0x200000001LL;
        }
        else
        {
          *a1 = v19;
          a1[1] = 0x200000000LL;
        }
        return a1;
      case '7':
        v13 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
        v14 = &a2[-v13];
        if ( (a2[7] & 0x40) != 0 )
          v14 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        v15 = a1 + 2;
        v16 = &v14[v13];
        LODWORD(v17) = 0;
        *a1 = a1 + 2;
        v18 = v13 >> 5;
        a1[1] = 0x200000000LL;
        if ( (unsigned __int64)v13 > 0x40 )
        {
          sub_C8D5F0((__int64)a1, v15, v13 >> 5, 8u, a5, a6);
          v17 = *((unsigned int *)a1 + 2);
          v15 = (_QWORD *)(*a1 + 8 * v17);
        }
        if ( v16 != v14 )
        {
          do
          {
            if ( v15 )
              *v15 = *(_QWORD *)v14;
            v14 += 32;
            ++v15;
          }
          while ( v14 != v16 );
          LODWORD(v17) = *((_DWORD *)a1 + 2);
        }
        *((_DWORD *)a1 + 2) = v17 + v18;
        return a1;
      case '8':
        v10 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
LABEL_11:
        a1[2] = v10;
        a1[1] = 0x200000001LL;
        *a1 = a1 + 2;
        return a1;
      case '9':
        if ( (a2[7] & 0x40) != 0 )
          v11 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v11 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v12 = *((_QWORD *)v11 + 8);
        a1[2] = *((_QWORD *)v11 + 4);
        *a1 = a1 + 2;
        a1[3] = v12;
        a1[1] = 0x200000002LL;
        return a1;
      case '@':
        break;
      default:
        BUG();
    }
  }
  *a1 = a1 + 2;
  a1[1] = 0x200000000LL;
  return a1;
}
