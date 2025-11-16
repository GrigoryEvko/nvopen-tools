// Function: sub_2FF9F90
// Address: 0x2ff9f90
//
__int64 __fastcall sub_2FF9F90(__int64 a1, int a2, int a3, unsigned int a4, unsigned __int64 a5, unsigned int a6)
{
  unsigned int v6; // r12d
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // eax
  unsigned int v14; // esi
  __int64 v15; // r9
  __int64 v16; // r15
  int v17; // ebx
  __int64 v18; // rax
  int v19; // r11d
  unsigned int v20; // esi
  int v21; // ebx
  __int64 v22; // rax
  int v23; // r11d
  __int64 v24; // rdi
  __int64 (*v25)(); // rax
  char v26; // al
  unsigned int v27; // [rsp+4h] [rbp-5Ch]
  unsigned int v28; // [rsp+8h] [rbp-58h]
  unsigned int v30; // [rsp+18h] [rbp-48h]
  char v31; // [rsp+18h] [rbp-48h]
  unsigned __int8 v33; // [rsp+27h] [rbp-39h] BYREF
  unsigned int v34; // [rsp+28h] [rbp-38h] BYREF
  _DWORD v35[13]; // [rsp+2Ch] [rbp-34h] BYREF

  if ( !*(_DWORD *)(a1 + 64) )
    return 0;
  LOBYTE(v11) = sub_2FF9BD0((_QWORD *)a1, a5, a4);
  v6 = v11;
  if ( !(_BYTE)v11 )
    return 0;
  v12 = sub_2FF8D40(a2, a1 + 240);
  if ( v12 )
  {
    v27 = v12;
    v30 = sub_2FF8D40(a3, a1 + 208);
    v13 = sub_2FF8D40(a4, a1 + 208);
    if ( !v30 )
    {
      if ( !v13 )
        goto LABEL_8;
      if ( sub_2FF8810(a1, v13, v27) )
        return v6;
      return 0;
    }
    v28 = v13;
    v26 = sub_2FF8810(a1, v30, v27);
    if ( !v28 )
    {
      if ( !v26 )
        return v6;
      return 0;
    }
    v31 = v26;
    if ( sub_2FF8810(a1, v28, v27) )
    {
      if ( !v31 )
        return v6;
      goto LABEL_8;
    }
    if ( v31 )
      return 0;
  }
LABEL_8:
  v34 = 0;
  if ( !(unsigned __int8)sub_2FF90F0(a1, a4, a6, &v34) )
    return 0;
  v35[0] = 0;
  if ( (unsigned __int8)sub_2FF90F0(a1, a3, a6, v35) )
  {
    v14 = a4;
    if ( (int)qword_502A528 > 0 )
    {
      v15 = *(_QWORD *)(a1 + 72);
      v16 = *(_QWORD *)(a1 + 32);
      v17 = 0;
      do
      {
        v18 = sub_2FF8E70(v16, v14, v15);
        if ( !v18 || *(_WORD *)(v18 + 68) != 20 )
          break;
        v14 = *(_DWORD *)(*(_QWORD *)(v18 + 32) + 48LL);
        if ( v14 == a2 )
          return v6;
        ++v17;
      }
      while ( v19 != v17 );
      v20 = a3;
      v21 = 0;
      do
      {
        v22 = sub_2FF8E70(v16, v20, v15);
        if ( !v22 || *(_WORD *)(v22 + 68) != 20 )
          break;
        v20 = *(_DWORD *)(*(_QWORD *)(v22 + 32) + 48LL);
        if ( v20 == a2 )
          return 0;
        ++v21;
      }
      while ( v23 != v21 );
    }
    v24 = *(_QWORD *)(a1 + 8);
    v25 = *(__int64 (**)())(*(_QWORD *)v24 + 272LL);
    if ( v25 != sub_2FDC4A0
      && ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, unsigned __int8 *))v25)(v24, a5, &v33) )
    {
      return v33;
    }
    if ( !v35[0] || v35[0] >= v34 )
      return 0;
  }
  return v6;
}
