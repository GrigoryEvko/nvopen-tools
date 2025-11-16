// Function: sub_2EA6A70
// Address: 0x2ea6a70
//
__int64 __fastcall sub_2EA6A70(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 (*v7)(); // rax
  __int64 v8; // r8
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rsi
  __int64 (*v12)(); // rax
  unsigned __int8 v13; // al
  char v14; // al
  __int64 v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 v19; // rax
  __int64 (*v20)(); // rax
  __int64 (*v21)(); // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 32LL);
  v5 = *(_QWORD *)(v4 + 16);
  v24 = *(_QWORD *)(v4 + 32);
  v22 = 0;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 200LL))(v5);
  v7 = *(__int64 (**)())(*(_QWORD *)v5 + 128LL);
  if ( v7 != sub_2DAC790 )
    v22 = ((__int64 (__fastcall *)(__int64))v7)(v5);
  v8 = *(_QWORD *)(a2 + 32);
  v9 = v8 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v9 == v8 )
    return 1;
  v10 = *(_QWORD *)(a2 + 32);
  while ( 1 )
  {
    if ( *(_BYTE *)v10 )
      goto LABEL_24;
    v11 = *(unsigned int *)(v10 + 8);
    if ( !(_DWORD)v11 || (_DWORD)v11 == a3 )
      goto LABEL_24;
    if ( (unsigned int)(v11 - 1) > 0x3FFFFFFE )
      break;
    v12 = *(__int64 (**)())(*(_QWORD *)v6 + 168LL);
    if ( v12 == sub_2EA3FB0
      || (LODWORD(v11) = *(_DWORD *)(v10 + 8), !((unsigned __int8 (__fastcall *)(__int64))v12)(v6)) )
    {
      v13 = *(_BYTE *)(v10 + 3);
      if ( (v13 & 0x10) != 0 )
      {
        if ( (((v13 & 0x10) != 0) & (v13 >> 6)) == 0 )
          return 0;
        v11 = (unsigned int)v11;
        if ( (unsigned __int8)sub_2E31DD0(**(_QWORD **)(a1 + 32), v11, -1, -1) )
          return 0;
        break;
      }
      if ( !(unsigned __int8)sub_2EA6910(a1, v11) )
      {
        v21 = *(__int64 (**)())(*(_QWORD *)v6 + 200LL);
        v19 = sub_2E88D60(a2);
        if ( v21 == sub_2E4EE50
          || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))v21)(v6, (unsigned int)v11, v19) )
        {
          v20 = *(__int64 (**)())(*(_QWORD *)v22 + 32LL);
          if ( v20 == sub_2E4EE60 || !((unsigned __int8 (__fastcall *)(__int64, __int64))v20)(v22, v10) )
            return 0;
        }
      }
LABEL_24:
      v10 += 40;
      if ( v9 == v10 )
        return 1;
    }
    else
    {
      v10 += 40;
      if ( v9 == v10 )
        return 1;
    }
  }
  v14 = *(_BYTE *)(v10 + 4);
  if ( (v14 & 1) != 0 || (v14 & 2) != 0 || (*(_BYTE *)(v10 + 3) & 0x10) != 0 && (*(_DWORD *)v10 & 0xFFF00) == 0 )
    goto LABEL_24;
  v15 = *(_QWORD *)(sub_2EBEE10(v24, v11) + 24);
  if ( !*(_BYTE *)(a1 + 84) )
  {
    if ( sub_C8CA60(a1 + 56, v15) )
      return 0;
    goto LABEL_24;
  }
  v16 = *(_QWORD **)(a1 + 64);
  v17 = &v16[*(unsigned int *)(a1 + 76)];
  if ( v16 == v17 )
    goto LABEL_24;
  while ( v15 != *v16 )
  {
    if ( v17 == ++v16 )
      goto LABEL_24;
  }
  return 0;
}
