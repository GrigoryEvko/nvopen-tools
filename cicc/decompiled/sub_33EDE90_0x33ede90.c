// Function: sub_33EDE90
// Address: 0x33ede90
//
_QWORD *__fastcall sub_33EDE90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v4; // r13
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r14
  __int64 (*v10)(); // rax
  char v11; // r9
  __int64 (*v12)(); // rax
  int v13; // eax
  __int64 v14; // r14
  int v15; // r13d
  __int64 v16; // rdi
  unsigned int v17; // esi
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int); // rax
  int v19; // edx
  unsigned __int16 v20; // ax
  char v22; // [rsp+8h] [rbp-28h]

  v4 = a4;
  v6 = *(_QWORD *)(a1 + 40);
  v7 = 0;
  v22 = a3;
  v8 = *(_QWORD *)(v6 + 16);
  v9 = *(_QWORD *)(v6 + 48);
  v10 = *(__int64 (**)())(*(_QWORD *)v8 + 136LL);
  if ( v10 != sub_2DD19D0 )
    v7 = ((__int64 (__fastcall *)(__int64))v10)(v8);
  v11 = 0;
  if ( v22 )
  {
    v12 = *(__int64 (**)())(*(_QWORD *)v7 + 328LL);
    if ( v12 != sub_2FDBCD0 )
      v11 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD))v12)(v7, a2, a3, a4, v8, 0);
  }
  v13 = sub_2E77BD0(v9, a2, v4, 0, 0, v11);
  v14 = *(_QWORD *)(a1 + 16);
  v15 = v13;
  v16 = sub_2E79000(*(__int64 **)(a1 + 40));
  v17 = *(_DWORD *)(v16 + 4);
  v18 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v14 + 32LL);
  if ( v18 == sub_2D42F30 )
  {
    v19 = sub_AE2980(v16, v17)[1];
    v20 = 2;
    if ( v19 != 1 )
    {
      v20 = 3;
      if ( v19 != 2 )
      {
        v20 = 4;
        if ( v19 != 4 )
        {
          v20 = 5;
          if ( v19 != 8 )
          {
            v20 = 6;
            if ( v19 != 16 )
            {
              v20 = 7;
              if ( v19 != 32 )
              {
                v20 = 8;
                if ( v19 != 64 )
                  v20 = 9 * (v19 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v20 = v18(v14, v16, v17);
  }
  return sub_33EDBD0((_QWORD *)a1, v15, v20, 0, 0);
}
