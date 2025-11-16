// Function: sub_35AA5C0
// Address: 0x35aa5c0
//
__int64 __fastcall sub_35AA5C0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 (*v3)(); // rax
  unsigned int v4; // r14d
  __int64 v5; // r13
  __int64 v6; // r15
  int v7; // ebx
  __int64 v8; // r14
  __int64 (*v9)(); // rdx
  void (__fastcall *v10)(__int64, _QWORD); // rax
  __int64 v11; // rax
  void (*v12)(); // rdx
  __int64 (*v13)(); // rax
  void (*v15)(); // rax
  unsigned int v16; // r13d
  __int64 v17; // rax
  void (__fastcall *v18)(__int64, _QWORD); // rdx
  int v19; // ebx
  void (*v20)(); // rax
  __int64 (*(__fastcall *v21)(__int64))(void); // rdx
  __int64 v22; // [rsp-50h] [rbp-50h]
  __int64 v23; // [rsp-48h] [rbp-48h]
  __int64 v24; // [rsp-40h] [rbp-40h]

  v2 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 128LL);
  if ( v2 == sub_2DAC790 )
    BUG();
  v23 = v2();
  v3 = *(__int64 (**)())(*(_QWORD *)v23 + 1056LL);
  if ( v3 != sub_2FDC740 )
  {
    v5 = ((__int64 (__fastcall *)(__int64, __int64))v3)(v23, a2);
    if ( v5 )
    {
      v4 = 0;
      v24 = *(_QWORD *)(a2 + 328);
      if ( v24 == a2 + 320 )
        goto LABEL_21;
      while ( 1 )
      {
        v6 = *(_QWORD *)(v24 + 56);
        v22 = v24 + 48;
        if ( v6 == v24 + 48 )
          goto LABEL_20;
        v7 = v4;
        v8 = v5;
        do
        {
          while ( 1 )
          {
            v9 = *(__int64 (**)())(*(_QWORD *)v8 + 64LL);
            if ( v9 == sub_2F8E400 )
            {
              v10 = *(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v8 + 104LL);
              if ( (char *)v10 != (char *)&loc_2F8E710 )
                v10(v8, 0);
              goto LABEL_12;
            }
            v16 = ((__int64 (__fastcall *)(__int64, __int64))v9)(v8, v6);
            v17 = *(_QWORD *)v8;
            v18 = *(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v8 + 104LL);
            if ( (char *)v18 == (char *)&loc_2F8E710 )
            {
              if ( v16 )
              {
                v19 = 0;
                while ( 1 )
                {
                  v21 = *(__int64 (*(__fastcall **)(__int64))(void))(v17 + 96);
                  if ( v21 == sub_2F39570 )
                  {
                    v20 = *(void (**)())(v17 + 80);
                    if ( v20 != nullsub_1620 )
                      ((void (__fastcall *)(__int64))v20)(v8);
                    if ( v16 == ++v19 )
                    {
LABEL_38:
                      v7 = 1;
                      (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v23 + 896LL))(
                        v23,
                        v24,
                        v6,
                        v16);
                      goto LABEL_13;
                    }
                  }
                  else
                  {
                    ++v19;
                    v21(v8);
                    if ( v16 == v19 )
                      goto LABEL_38;
                  }
                  v17 = *(_QWORD *)v8;
                }
              }
LABEL_12:
              (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v23 + 896LL))(v23, v24, v6, 0);
              goto LABEL_13;
            }
            v18(v8, v16);
            (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v23 + 896LL))(v23, v24, v6, v16);
            if ( v16 )
              v7 = 1;
LABEL_13:
            v11 = *(_QWORD *)v8;
            v12 = *(void (**)())(*(_QWORD *)v8 + 48LL);
            if ( v12 != nullsub_1663 )
            {
              ((void (__fastcall *)(__int64, __int64))v12)(v8, v6);
              v11 = *(_QWORD *)v8;
            }
            v13 = *(__int64 (**)())(v11 + 16);
            if ( v13 != sub_2F39220 )
            {
              if ( ((unsigned __int8 (__fastcall *)(__int64))v13)(v8) )
              {
                v15 = *(void (**)())(*(_QWORD *)v8 + 80LL);
                if ( v15 != nullsub_1620 )
                  ((void (__fastcall *)(__int64))v15)(v8);
              }
            }
            if ( !v6 )
              BUG();
            if ( (*(_BYTE *)v6 & 4) == 0 )
              break;
            v6 = *(_QWORD *)(v6 + 8);
            if ( v22 == v6 )
              goto LABEL_19;
          }
          while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
            v6 = *(_QWORD *)(v6 + 8);
          v6 = *(_QWORD *)(v6 + 8);
        }
        while ( v22 != v6 );
LABEL_19:
        v5 = v8;
        v4 = v7;
LABEL_20:
        v24 = *(_QWORD *)(v24 + 8);
        if ( a2 + 320 == v24 )
        {
LABEL_21:
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
          return v4;
        }
      }
    }
  }
  return 0;
}
