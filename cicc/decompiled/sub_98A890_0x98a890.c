// Function: sub_98A890
// Address: 0x98a890
//
__int64 __fastcall sub_98A890(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        unsigned __int8 a7,
        char a8)
{
  __int64 v8; // r15
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // r8
  __int64 v18; // r15
  int v19; // eax
  __int64 result; // rax
  unsigned int *v21; // rsi
  __int64 v22; // r14
  __int64 v23; // r12
  unsigned int *v24; // rax
  __int64 v25; // r12
  unsigned int *v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // r12
  __int64 v29; // rdi
  __int64 v30; // [rsp-10h] [rbp-B0h]
  __int64 v31; // [rsp-8h] [rbp-A8h]
  __int64 v32; // [rsp+8h] [rbp-98h]
  __int64 v33; // [rsp+10h] [rbp-90h]
  __int64 v34; // [rsp+20h] [rbp-80h]
  __int64 v36; // [rsp+38h] [rbp-68h]
  __int64 v37; // [rsp+38h] [rbp-68h]
  __int64 v38[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v39; // [rsp+50h] [rbp-50h]
  char v40; // [rsp+60h] [rbp-40h]
  char v41; // [rsp+61h] [rbp-3Fh]

  v8 = a6;
  v10 = a2;
  v11 = a4;
  if ( *(_BYTE *)(a3 + 8) == 15 )
  {
    v12 = *(unsigned int *)(a3 + 12);
    if ( (_DWORD)v12 )
    {
      v14 = *(unsigned int *)(a4 + 8);
      v34 = v12;
      v15 = 0;
      v33 = a4 + 16;
      v16 = a3;
      v17 = v14 + 1;
      v18 = a2;
      v19 = 0;
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        goto LABEL_7;
      while ( 1 )
      {
        v36 = a6;
        *(_DWORD *)(*(_QWORD *)v11 + 4 * v14) = v19;
        ++*(_DWORD *)(v11 + 8);
        result = sub_98A890(a1, v18, *(_QWORD *)(*(_QWORD *)(v16 + 16) + 8 * v15), v11, a5, a6, a7, a8);
        a6 = v36;
        v14 = (unsigned int)(*(_DWORD *)(v11 + 8) - 1);
        *(_DWORD *)(v11 + 8) = v14;
        a4 = v30;
        if ( !result )
          break;
        if ( v34 == ++v15 )
          return result;
        v17 = v14 + 1;
        v18 = result;
        v19 = v15;
        if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 12) )
        {
LABEL_7:
          v32 = a6;
          sub_C8D5F0(v11, v33, v17, 4);
          v14 = *(unsigned int *)(v11 + 8);
          a6 = v32;
          v19 = v15;
        }
      }
      v28 = v18;
      v8 = v36;
      if ( a2 == v28 )
      {
        v10 = 0;
      }
      else
      {
        do
        {
          v29 = v28;
          v28 = *(_QWORD *)(v28 - 64);
          sub_B43D60(v29, v31, v14, a4);
        }
        while ( a2 != v28 );
        v14 = *(unsigned int *)(v11 + 8);
        v10 = 0;
      }
      goto LABEL_9;
    }
    if ( a2 )
      return a2;
  }
  v14 = *(unsigned int *)(a4 + 8);
LABEL_9:
  v21 = *(unsigned int **)v11;
  LOBYTE(v39) = 0;
  v22 = sub_98A4C0(a1, v21, v14, a4, a5, a6, v38[0], v38[1], v39);
  if ( !v22 )
    return 0;
  v23 = *(unsigned int *)(v11 + 8);
  v38[0] = (__int64)"tmp";
  v24 = *(unsigned int **)v11;
  v25 = v23 - a5;
  v41 = 1;
  v26 = &v24[a5];
  v40 = 3;
  result = sub_BD2C40(104, unk_3F148BC);
  if ( result )
  {
    v27 = a7;
    v37 = result;
    BYTE1(v27) = a8;
    sub_B44260(result, *(_QWORD *)(v10 + 8), 65, 2, v8, v27);
    *(_QWORD *)(v37 + 80) = 0x400000000LL;
    *(_QWORD *)(v37 + 72) = v37 + 88;
    sub_B4FD20(v37, v10, v22, v26, v25, v38);
    return v37;
  }
  return result;
}
