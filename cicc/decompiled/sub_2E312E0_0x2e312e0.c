// Function: sub_2E312E0
// Address: 0x2e312e0
//
__int64 __fastcall sub_2E312E0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rcx
  unsigned int v6; // r14d
  __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  __int64 v11; // r12
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  __int64 (*v15)(); // rax
  char v16; // al
  __int64 v17; // [rsp+8h] [rbp-38h]

  v5 = 0;
  v6 = a3;
  v8 = a2;
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 128LL);
  if ( v10 != sub_2DAC790 )
    v5 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v10)(v9, a2, a3, 0);
  v11 = a1 + 48;
  if ( v11 != a2 )
  {
    v12 = 508025;
    while ( 1 )
    {
      v13 = *(unsigned __int16 *)(v8 + 68);
      if ( (unsigned __int16)v13 > 0x12u )
      {
        if ( (_WORD)v13 == 68 || (_WORD)v13 == 24 && a4 )
          goto LABEL_9;
      }
      else if ( _bittest64(&v12, v13) )
      {
        goto LABEL_9;
      }
      v15 = *(__int64 (**)())(*(_QWORD *)v5 + 1336LL);
      if ( v15 == sub_2E2F9B0 )
        return v8;
      v17 = v5;
      v16 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v15)(v5, v8, v6);
      v5 = v17;
      if ( !v16 )
        return v8;
LABEL_9:
      if ( (*(_BYTE *)v8 & 4) != 0 )
      {
        v8 = *(_QWORD *)(v8 + 8);
        if ( v8 == v11 )
          return v11;
      }
      else
      {
        while ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
        v8 = *(_QWORD *)(v8 + 8);
        if ( v8 == v11 )
          return v11;
      }
    }
  }
  return v11;
}
