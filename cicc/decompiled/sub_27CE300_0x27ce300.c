// Function: sub_27CE300
// Address: 0x27ce300
//
char __fastcall sub_27CE300(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 **v4; // rdx
  unsigned __int8 *v5; // rbx
  int v6; // eax
  char result; // al
  unsigned __int8 *v8; // rdx
  int v9; // eax
  unsigned __int8 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // r13d
  __int64 v14; // rcx
  int v15; // r14d
  int v16; // edi
  unsigned __int8 *v17; // rcx
  int v18; // edi
  int v19; // edi

  if ( (a1[7] & 0x40) == 0 )
  {
    v8 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v5 = *(unsigned __int8 **)v8;
    v6 = **(unsigned __int8 **)v8;
    if ( (unsigned __int8)v6 <= 0x1Cu )
      goto LABEL_3;
LABEL_7:
    v9 = v6 - 29;
    goto LABEL_8;
  }
  v4 = (unsigned __int8 **)*((_QWORD *)a1 - 1);
  v5 = *v4;
  v6 = **v4;
  if ( (unsigned __int8)v6 > 0x1Cu )
    goto LABEL_7;
LABEL_3:
  if ( (_BYTE)v6 != 5 )
    return 0;
  v9 = *((unsigned __int16 *)v5 + 1);
LABEL_8:
  if ( v9 != 47 )
    return 0;
  if ( (v5[7] & 0x40) != 0 )
    v10 = (unsigned __int8 *)*((_QWORD *)v5 - 1);
  else
    v10 = &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
  v11 = *(_QWORD *)(*(_QWORD *)v10 + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
    v11 = **(_QWORD **)(v11 + 16);
  v12 = *((_QWORD *)a1 + 1);
  v13 = *(_DWORD *)(v11 + 8) >> 8;
  v14 = v12;
  if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
    v14 = **(_QWORD **)(v12 + 16);
  v15 = *(_DWORD *)(v14 + 8) >> 8;
  v16 = *a1 <= 0x1Cu ? *((unsigned __int16 *)a1 + 1) : *a1 - 29;
  if ( !sub_B50750(v16, *((_QWORD *)v5 + 1), v12, a2) )
    return 0;
  v17 = (v5[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v5 - 1) : &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
  v18 = *v5;
  v19 = (unsigned __int8)v18 <= 0x1Cu ? *((unsigned __int16 *)v5 + 1) : v18 - 29;
  result = sub_B50750(v19, *(_QWORD *)(*(_QWORD *)v17 + 8LL), *((_QWORD *)v5 + 1), a2);
  if ( !result )
    return 0;
  if ( v15 != v13 )
    return sub_DF9B10(a3);
  return result;
}
