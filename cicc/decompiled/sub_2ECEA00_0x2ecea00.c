// Function: sub_2ECEA00
// Address: 0x2ecea00
//
__int64 __fastcall sub_2ECEA00(__int64 a1, _QWORD *a2)
{
  _DWORD *v3; // rdi
  __int64 (*v4)(); // rax
  int v5; // eax
  int v6; // edx
  _DWORD *v7; // rdi
  unsigned int v8; // r13d
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned __int16 *v13; // r12
  unsigned int v14; // r15d
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int16 *v17; // [rsp+8h] [rbp-38h]

  v3 = *(_DWORD **)(a1 + 152);
  if ( v3[2] )
  {
    v4 = *(__int64 (**)())(*(_QWORD *)v3 + 24LL);
    if ( v4 != sub_2EC0B50 )
    {
      if ( ((unsigned int (__fastcall *)(_DWORD *, _QWORD *, _QWORD))v4)(v3, a2, 0) )
        return 1;
    }
  }
  v5 = sub_2FF7F40(*(_QWORD *)(a1 + 8), *a2, 0);
  v6 = *(_DWORD *)(a1 + 168);
  if ( v6 )
  {
    v7 = *(_DWORD **)(a1 + 8);
    if ( (unsigned int)(v6 + v5) > *v7 )
      return 1;
    if ( *(_DWORD *)(a1 + 24) != 1 )
    {
LABEL_6:
      if ( !(unsigned __int8)sub_2FF7ED0(v7, *a2, 0) )
        goto LABEL_7;
      return 1;
    }
    if ( (unsigned __int8)sub_2FF7E60(v7, *a2, 0) )
      return 1;
    if ( *(_DWORD *)(a1 + 24) != 1 )
    {
      v7 = *(_DWORD **)(a1 + 8);
      goto LABEL_6;
    }
  }
LABEL_7:
  if ( !(unsigned __int8)sub_2FF7B70(*(_QWORD *)(a1 + 8)) )
    return 0;
  v8 = *((_BYTE *)a2 + 249) >> 7;
  if ( *((char *)a2 + 249) >= 0 )
    return 0;
  v10 = a2[2];
  if ( !v10 )
  {
    v15 = *(_QWORD *)a1 + 600LL;
    if ( (unsigned __int8)sub_2FF7B70(v15) )
    {
      v16 = sub_2FF7DB0(v15, *a2);
      a2[2] = v16;
      v10 = v16;
    }
    else
    {
      v10 = a2[2];
    }
  }
  v11 = *(unsigned __int16 *)(v10 + 2);
  v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 192LL) + 176LL);
  v13 = (unsigned __int16 *)(v12 + 6 * v11);
  v17 = (unsigned __int16 *)(v12 + 6 * (v11 + *(unsigned __int16 *)(v10 + 4)));
  if ( v13 != v17 )
  {
    while ( 1 )
    {
      v14 = v13[1];
      if ( *(_DWORD *)(a1 + 164) < (unsigned int)sub_2ECE820((_QWORD *)a1, v10, *v13, v14, v13[2]) )
        break;
      v13 += 3;
      if ( v17 == v13 )
        return 0;
    }
    if ( *(_DWORD *)(a1 + 712) >= v14 )
      v14 = *(_DWORD *)(a1 + 712);
    *(_DWORD *)(a1 + 712) = v14;
  }
  else
  {
    return 0;
  }
  return v8;
}
