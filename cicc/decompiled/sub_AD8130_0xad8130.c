// Function: sub_AD8130
// Address: 0xad8130
//
__int64 __fastcall sub_AD8130(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v5; // r12
  __int64 v6; // rax
  _BYTE *v7; // rdi
  __int64 v9; // rdx
  int v10; // eax
  int v11; // r14d
  unsigned int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdi
  unsigned __int8 *v19; // rax

  v5 = (unsigned __int8 *)a1;
  if ( *(_BYTE *)a1 == 18 )
    goto LABEL_2;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(unsigned __int8 *)(v9 + 8);
  if ( (_BYTE)v10 == 17 )
  {
    v11 = *(_DWORD *)(v9 + 32);
    if ( !v11 )
      return 1;
    v12 = 0;
    while ( 1 )
    {
      v13 = sub_AD69F0(v5, v12);
      if ( !v13 || *(_BYTE *)v13 != 18 )
        break;
      v17 = v13 + 24;
      if ( !(*(_QWORD *)(v13 + 24) == sub_C33340(v5, v12, v14, v15, v16)
           ? sub_C40A10(v17, 0)
           : (unsigned __int8)sub_C408D0(v17, 0)) )
        break;
      if ( v11 == ++v12 )
        return 1;
    }
    return 0;
  }
  if ( (unsigned int)(v10 - 17) > 1 )
    return 0;
  a2 = 0;
  v19 = sub_AD7630(a1, 0, v9);
  v5 = v19;
  if ( !v19 || *v19 != 18 )
    return 0;
LABEL_2:
  v6 = sub_C33340(a1, a2, a3, a4, a5);
  v7 = v5 + 24;
  if ( *((_QWORD *)v5 + 3) == v6 )
    return sub_C40A10(v7, 0);
  else
    return sub_C408D0(v7, 0);
}
