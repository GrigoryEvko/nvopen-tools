// Function: sub_359CAC0
// Address: 0x359cac0
//
void __fastcall sub_359CAC0(__int64 *a1, __int64 a2, int a3, int a4, _QWORD *a5, __int64 a6)
{
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 i; // r15
  int v11; // r14d
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // rsi
  int v15; // eax
  int v16; // r13d
  __int64 v17; // rdi
  int v18; // edx
  int v19; // r13d
  __int32 v20; // r13d
  __int64 v21; // r9
  __int64 v22; // rdi
  __int32 v23; // r8d
  unsigned __int8 *v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int32 v29; // [rsp+4h] [rbp-ACh]
  __int64 v30; // [rsp+8h] [rbp-A8h]
  __int64 v33; // [rsp+20h] [rbp-90h]
  int v34; // [rsp+20h] [rbp-90h]
  __int32 v38; // [rsp+40h] [rbp-70h] BYREF
  int v39; // [rsp+44h] [rbp-6Ch] BYREF
  unsigned __int8 *v40; // [rsp+48h] [rbp-68h] BYREF
  __int64 v41[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v42[10]; // [rsp+60h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a2 + 32);
  v8 = v7 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  v9 = v7 + 40LL * (unsigned int)sub_2E88FE0(a2);
  if ( v8 != v9 )
  {
    for ( i = v9; v8 != i; i += 40 )
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)i )
        {
          v11 = *(_DWORD *)(i + 8);
          if ( v11 < 0 )
          {
            v12 = sub_2EBEE10(a1[3], v11);
            v13 = v12;
            if ( v12 )
            {
              v14 = a1[6];
              if ( v14 == *(_QWORD *)(v12 + 24) )
                break;
            }
          }
        }
LABEL_3:
        i += 40;
        if ( v8 == i )
          return;
      }
      v15 = *(unsigned __int16 *)(v12 + 68);
      v38 = 0;
      v39 = v11;
      if ( v15 == 68 || (v16 = 0, !v15) )
      {
        LODWORD(v42[0]) = 0;
        v16 = 1;
        sub_3598310(v13, v14, &v38, v42);
        v17 = a1[3];
        v39 = v42[0];
        v13 = sub_2EBEE10(v17, v42[0]);
      }
      v18 = a3 + v16 - sub_3598DB0(*a1, v13);
      v19 = v18;
      if ( v18 <= a4 && (v33 = *a5 + 32LL * (a4 - v18), sub_359BBF0(v33, &v39)) )
      {
        v20 = *sub_2FFAE70(v33, &v39);
      }
      else if ( a6 )
      {
        v20 = *sub_2FFAE70(*(_QWORD *)a6 + 32 * (*(unsigned int *)(a6 + 8) - (__int64)(v19 - a4)), &v39);
      }
      else
      {
        v20 = v38;
      }
      if ( !sub_2EBE590(
              a1[3],
              v20,
              *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (v11 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
              0) )
      {
        v23 = sub_2EC06C0(
                a1[3],
                *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (v11 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                byte_3F871B3,
                0,
                16LL * (v11 & 0x7FFFFFFF),
                v21);
        v24 = *(unsigned __int8 **)(a2 + 56);
        v25 = *(_QWORD *)(a1[4] + 8) - 800LL;
        v40 = v24;
        if ( v24 )
        {
          v29 = v23;
          v30 = v25;
          sub_B96E90((__int64)&v40, (__int64)v24, 1);
          v25 = v30;
          v23 = v29;
          v42[0] = (__int64)v40;
          if ( v40 )
          {
            sub_B976B0((__int64)&v40, v40, (__int64)v42);
            v40 = 0;
            v23 = v29;
            v25 = v30;
          }
        }
        else
        {
          v42[0] = 0;
        }
        v26 = a1[6];
        v34 = v23;
        v42[1] = 0;
        v42[2] = 0;
        v27 = sub_2F2A600(v26, a2, v42, v25, v23);
        v41[1] = v28;
        v41[0] = (__int64)v27;
        sub_3598AB0(v41, v20, 0, 0);
        sub_9C6650(v42);
        sub_9C6650(&v40);
        sub_2EAB0C0(i, v34);
        goto LABEL_3;
      }
      v22 = i;
      sub_2EAB0C0(v22, v20);
    }
  }
}
