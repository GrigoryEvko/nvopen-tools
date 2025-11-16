// Function: sub_23D8620
// Address: 0x23d8620
//
__int64 __fastcall sub_23D8620(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned __int8 *v7; // r12
  __int64 v8; // r12
  unsigned __int64 v9; // rdx
  unsigned __int8 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned __int8 *v13; // r12
  unsigned __int8 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int8 *v17; // r12
  __int64 v18; // r13
  unsigned __int8 *v19; // rdx
  unsigned __int8 *v20; // r13
  unsigned __int8 *v21; // r12
  __int64 v22; // r14
  unsigned __int64 v23; // rdx

  result = (unsigned int)*a1 - 42;
  switch ( *a1 )
  {
    case '*':
    case ',':
    case '.':
    case '0':
    case '3':
    case '6':
    case '7':
    case '8':
    case '9':
    case ':':
    case ';':
    case '[':
      if ( (a1[7] & 0x40) != 0 )
        v10 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v10 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v11 = *(unsigned int *)(a2 + 8);
      v12 = *(_QWORD *)v10;
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v11 + 1, 8u, a5, a6);
        v11 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v11) = v12;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
      if ( (a1[7] & 0x40) != 0 )
        v13 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v13 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v9 = result + 1;
      v8 = *((_QWORD *)v13 + 4);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        goto LABEL_14;
      goto LABEL_5;
    case 'C':
    case 'D':
    case 'E':
      return result;
    case 'T':
      v18 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
      v19 = &a1[-v18];
      if ( (a1[7] & 0x40) != 0 )
        v19 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      v20 = &v19[v18];
      if ( v20 == v19 )
        return result;
      result = *(unsigned int *)(a2 + 8);
      v21 = v19;
      v22 = *(_QWORD *)v19;
      v23 = result + 1;
      if ( result + 1 <= (unsigned __int64)*(unsigned int *)(a2 + 12) )
        goto LABEL_27;
      break;
    case 'V':
      if ( (a1[7] & 0x40) != 0 )
        v14 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v14 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v15 = *(unsigned int *)(a2 + 8);
      v16 = *((_QWORD *)v14 + 4);
      if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v15 + 1, 8u, a5, a6);
        v15 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v15) = v16;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
      if ( (a1[7] & 0x40) != 0 )
        v17 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v17 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v9 = result + 1;
      v8 = *((_QWORD *)v17 + 8);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        goto LABEL_14;
      goto LABEL_5;
    case 'Z':
      if ( (a1[7] & 0x40) != 0 )
        v7 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v7 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      result = *(unsigned int *)(a2 + 8);
      v8 = *(_QWORD *)v7;
      v9 = result + 1;
      if ( result + 1 <= (unsigned __int64)*(unsigned int *)(a2 + 12) )
        goto LABEL_5;
LABEL_14:
      sub_C8D5F0(a2, (const void *)(a2 + 16), v9, 8u, a5, a6);
      result = *(unsigned int *)(a2 + 8);
LABEL_5:
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v8;
      ++*(_DWORD *)(a2 + 8);
      return result;
    default:
      BUG();
  }
LABEL_29:
  sub_C8D5F0(a2, (const void *)(a2 + 16), v23, 8u, a5, a6);
  result = *(unsigned int *)(a2 + 8);
LABEL_27:
  while ( 1 )
  {
    v21 += 32;
    *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v22;
    result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = result;
    if ( v20 == v21 )
      return result;
    v23 = result + 1;
    v22 = *(_QWORD *)v21;
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      goto LABEL_29;
  }
}
