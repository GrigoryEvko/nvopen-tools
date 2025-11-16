// Function: sub_2FB0E40
// Address: 0x2fb0e40
//
__int64 __fastcall sub_2FB0E40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rax
  __int64 v9; // rdi
  __int64 (*v10)(); // rcx
  __int64 v11; // rdx
  char *v12; // rsi
  __int64 v13; // r12
  __int64 v14; // r12
  char *v15; // rax
  char *i; // rsi
  __int64 v18; // rsi

  v8 = *(_QWORD **)(a2 + 24);
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)a1 = v8;
  *(_QWORD *)(a1 + 24) = a4;
  v9 = v8[2];
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 128LL);
  v11 = 0;
  if ( v10 != sub_2DAC790 )
  {
    v11 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v10)(v9, a2, 0);
    v8 = *(_QWORD **)a1;
  }
  *(_QWORD *)(a1 + 32) = v11;
  v12 = (char *)(a1 + 72);
  *(_QWORD *)(a1 + 40) = 0;
  v13 = v8[13] - v8[12];
  *(_QWORD *)(a1 + 48) = a3;
  v14 = v13 >> 3;
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 64) = 0x800000000LL;
  if ( (_DWORD)v14 )
  {
    v15 = (char *)(a1 + 72);
    if ( (unsigned int)v14 > 8uLL )
    {
      sub_C8D5F0(a1 + 56, v12, (unsigned int)v14, 0x10u, a5, a6);
      v18 = *(_QWORD *)(a1 + 56);
      v15 = (char *)(v18 + 16LL * *(unsigned int *)(a1 + 64));
      for ( i = (char *)(16LL * (unsigned int)v14 + v18); i != v15; v15 += 16 )
      {
LABEL_6:
        if ( v15 )
        {
          *(_QWORD *)v15 = 0;
          *((_QWORD *)v15 + 1) = 0;
        }
      }
    }
    else
    {
      i = &v12[16 * (unsigned int)v14];
      if ( i != v15 )
        goto LABEL_6;
    }
    *(_DWORD *)(a1 + 64) = v14;
  }
  *(_DWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 208) = 0x800000000LL;
  *(_QWORD *)(a1 + 288) = 0x800000000LL;
  *(_QWORD *)(a1 + 624) = a1 + 640;
  *(_QWORD *)(a1 + 280) = a1 + 296;
  *(_QWORD *)(a1 + 632) = 0x600000000LL;
  *(_DWORD *)(a1 + 688) = 0;
  *(_DWORD *)(a1 + 696) = 0;
  *(_BYTE *)(a1 + 700) = 0;
  return 0x600000000LL;
}
