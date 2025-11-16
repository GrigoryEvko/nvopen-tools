// Function: sub_1EA9530
// Address: 0x1ea9530
//
__int64 __fastcall sub_1EA9530(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // r15
  __int64 (*v4)(); // rax
  __int64 v6; // r12
  __int64 i; // r14
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 (*v13)(); // rcx
  void (*v14)(); // rcx
  __int64 (*v15)(); // rax
  __int64 v16; // rax
  void (*v17)(); // rax
  __int64 v18; // rax
  int v19; // ebx
  __int64 v20; // r13
  __int64 v21; // r14
  void (*v22)(); // rax
  __int64 (*(__fastcall *v23)(__int64))(void); // rdx
  __int64 v24; // rax
  __int64 v25; // [rsp-48h] [rbp-48h]
  int v26; // [rsp-3Ch] [rbp-3Ch]

  v2 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 40LL);
  if ( v2 == sub_1D00B00 )
    BUG();
  v3 = v2();
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 776LL);
  if ( v4 == sub_1EA94D0 )
    return 0;
  v6 = ((__int64 (__fastcall *)(__int64, __int64))v4)(v3, a2);
  if ( !v6 )
    return 0;
  for ( i = *(_QWORD *)(a2 + 328); a2 + 320 != i; i = *(_QWORD *)(i + 8) )
  {
    v8 = *(_QWORD *)(i + 32);
    v25 = i + 24;
    if ( v8 != i + 24 )
    {
      v9 = v3;
      v10 = i;
      v11 = v9;
      do
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)v6;
          v13 = *(__int64 (**)())(*(_QWORD *)v6 + 64LL);
          if ( v13 != sub_1EA94F0 )
          {
            v26 = ((__int64 (__fastcall *)(__int64, __int64))v13)(v6, v8);
            if ( v26 )
            {
              v18 = v8;
              v19 = 0;
              v20 = v11;
              v21 = v18;
              do
              {
                v23 = *(__int64 (*(__fastcall **)(__int64))(void))(*(_QWORD *)v6 + 96LL);
                if ( v23 == sub_1D123B0 )
                {
                  v22 = *(void (**)())(*(_QWORD *)v6 + 80LL);
                  if ( v22 != nullsub_683 )
                    ((void (__fastcall *)(__int64))v22)(v6);
                }
                else
                {
                  v23(v6);
                }
                ++v19;
                (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v20 + 632LL))(v20, v10, v21);
              }
              while ( v19 != v26 );
              v24 = v21;
              v11 = v20;
              v8 = v24;
            }
            v12 = *(_QWORD *)v6;
          }
          v14 = *(void (**)())(v12 + 48);
          if ( v14 != nullsub_727 )
          {
            ((void (__fastcall *)(__int64, __int64))v14)(v6, v8);
            v12 = *(_QWORD *)v6;
          }
          v15 = *(__int64 (**)())(v12 + 16);
          if ( v15 != sub_1D00B80 )
          {
            if ( ((unsigned __int8 (__fastcall *)(__int64))v15)(v6) )
            {
              v17 = *(void (**)())(*(_QWORD *)v6 + 80LL);
              if ( v17 != nullsub_683 )
                ((void (__fastcall *)(__int64))v17)(v6);
            }
          }
          if ( !v8 )
            BUG();
          if ( (*(_BYTE *)v8 & 4) == 0 )
            break;
          v8 = *(_QWORD *)(v8 + 8);
          if ( v25 == v8 )
            goto LABEL_16;
        }
        while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
        v8 = *(_QWORD *)(v8 + 8);
      }
      while ( v25 != v8 );
LABEL_16:
      v16 = v11;
      i = v10;
      v3 = v16;
    }
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  return 1;
}
