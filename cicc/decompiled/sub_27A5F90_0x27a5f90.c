// Function: sub_27A5F90
// Address: 0x27a5f90
//
__int64 __fastcall sub_27A5F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, _DWORD *a6)
{
  __int64 v7; // r15
  __int64 v10; // rdi
  unsigned __int8 **v11; // rax
  __int64 v12; // r14
  char v13; // al
  int v15; // eax
  __int64 v16; // rdi
  __int64 v17; // rsi
  int v18; // r8d
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r11
  unsigned int v22; // r11d
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // rdi
  int v26; // eax
  int v27; // eax
  int v28; // ecx
  int v29; // ecx
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  unsigned __int8 *v34; // [rsp+28h] [rbp-38h]

  if ( a2 == a3 )
    return 1;
  v7 = *(_QWORD *)(a2 + 40);
  v31 = *(_QWORD *)(a3 + 40);
  v10 = *(_QWORD *)(a1 + 216);
  v30 = *(_QWORD *)(a4 + 64);
  v11 = (unsigned __int8 **)(a4 - 64);
  if ( *(_BYTE *)a4 == 26 )
    v11 = (unsigned __int8 **)(a4 - 32);
  v12 = *((_QWORD *)*v11 + 8);
  v34 = *v11;
  sub_B196A0(v10, v7, v12);
  if ( !v13 )
  {
    if ( v7 != v12 || v34 == *(unsigned __int8 **)(*(_QWORD *)(a1 + 248) + 128LL) || (unsigned int)*v34 - 26 > 1 )
      goto LABEL_6;
    v15 = *(_DWORD *)(a1 + 288);
    v16 = *((_QWORD *)v34 + 9);
    v17 = *(_QWORD *)(a1 + 272);
    if ( v15 )
    {
      v18 = v15 - 1;
      v19 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v20 = (__int64 *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( v16 == *v20 )
      {
LABEL_17:
        v22 = *((_DWORD *)v20 + 2);
      }
      else
      {
        v26 = 1;
        while ( v21 != -4096 )
        {
          v29 = v26 + 1;
          v19 = v18 & (v26 + v19);
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( v16 == *v20 )
            goto LABEL_17;
          v26 = v29;
        }
        v22 = 0;
      }
      v23 = v18 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = (__int64 *)(v17 + 16LL * v23);
      v25 = *v24;
      if ( a2 != *v24 )
      {
        v27 = 1;
        while ( v25 != -4096 )
        {
          v28 = v27 + 1;
          v23 = v18 & (v27 + v23);
          v24 = (__int64 *)(v17 + 16LL * v23);
          v25 = *v24;
          if ( a2 == *v24 )
            goto LABEL_19;
          v27 = v28;
        }
        return 0;
      }
LABEL_19:
      if ( *((_DWORD *)v24 + 2) > v22 )
      {
LABEL_6:
        if ( a5 == 3 )
        {
          if ( (unsigned __int8)sub_27A5D10(a1, a2, a4, a6) )
            return 0;
        }
        else if ( (unsigned __int8)sub_27A5AB0(a1, v7, v31, a6) )
        {
          return 0;
        }
        if ( v7 == v30 )
          sub_B196A0(*(_QWORD *)(a1 + 216), v12, v7);
        return 1;
      }
    }
  }
  return 0;
}
