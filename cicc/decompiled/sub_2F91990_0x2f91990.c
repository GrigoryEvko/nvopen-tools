// Function: sub_2F91990
// Address: 0x2f91990
//
unsigned __int64 __fastcall sub_2F91990(__int64 a1, unsigned __int64 a2, unsigned int a3)
{
  unsigned int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned int v11; // r13d
  unsigned __int64 result; // rax
  unsigned int v13; // edi
  unsigned int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // r9
  unsigned __int16 *v19; // rdx
  __int64 (*v20)(); // rcx
  unsigned int v21; // r10d
  char v22; // al
  __int64 v23; // rdi
  void (*v24)(); // rax
  __int64 *v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r13
  int v28; // eax
  int v29; // edx
  __int64 v30; // rax
  unsigned __int16 *v31; // rdi
  __int64 v32; // rsi
  unsigned __int16 *v33; // rax
  unsigned __int64 v34; // r8
  __int16 *v35; // [rsp+8h] [rbp-78h]
  unsigned int v36; // [rsp+10h] [rbp-70h]
  unsigned int v38; // [rsp+18h] [rbp-68h]
  unsigned int v39; // [rsp+28h] [rbp-58h]
  char v40; // [rsp+2Fh] [rbp-51h]
  int v41; // [rsp+3Ch] [rbp-44h] BYREF
  unsigned __int64 v42; // [rsp+40h] [rbp-40h] BYREF
  __int64 v43; // [rsp+48h] [rbp-38h]

  v40 = 0;
  v5 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 32LL) + 40LL * a3 + 8);
  v6 = *(_QWORD *)(a1 + 32);
  v7 = *(_QWORD *)(*(_QWORD *)a2 + 16LL);
  v8 = *(_QWORD *)(v6 + 16);
  if ( a3 >= *(unsigned __int16 *)(v7 + 2) )
    v40 = sub_3148160(v7, v5, 0) ^ 1;
  v9 = *(_QWORD *)(a1 + 24);
  v10 = *(_QWORD *)(v9 + 8);
  v11 = *(_DWORD *)(v10 + 24LL * v5 + 16) & 0xFFF;
  v35 = (__int16 *)(*(_QWORD *)(v9 + 56) + 2LL * (*(_DWORD *)(v10 + 24LL * v5 + 16) >> 12));
  result = a2 & 0xFFFFFFFFFFFFFFF9LL;
  while ( v35 )
  {
    result = v11;
    v13 = *(_DWORD *)(a1 + 1208);
    v14 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 1408) + 2LL * v11);
    if ( v14 >= v13 )
      goto LABEL_30;
    v15 = *(_QWORD *)(a1 + 1200);
    while ( 1 )
    {
      result = v14;
      v16 = v15 + 24LL * v14;
      if ( v11 == *(_DWORD *)(v16 + 12) )
      {
        v17 = *(unsigned int *)(v16 + 16);
        if ( (_DWORD)v17 != -1 && *(_DWORD *)(v15 + 24 * v17 + 20) == -1 )
          break;
      }
      v14 += 0x10000;
      if ( v13 <= v14 )
        goto LABEL_30;
    }
    if ( v14 != -1 )
    {
      v36 = v11;
      while ( 2 )
      {
        v26 = 24 * result;
        v25 = (__int64 *)(v15 + 24 * result);
        v27 = *v25;
        if ( a2 == *v25 )
          goto LABEL_21;
        v20 = *(__int64 (**)())(*(_QWORD *)a1 + 128LL);
        if ( v20 == sub_2EC0A10 )
        {
LABEL_24:
          v34 = *((unsigned int *)v25 + 2);
          v42 = 0;
          v43 = 0;
          if ( (v34 & 0x80000000) == 0LL )
          {
            *(_BYTE *)(a2 + 248) |= 0x40u;
            v18 = *(_QWORD *)v27;
            v19 = *(unsigned __int16 **)(*(_QWORD *)v27 + 16LL);
            v20 = *(__int64 (**)())(*(_QWORD *)v27 + 32LL);
            v21 = *((_DWORD *)v20 + 10 * (int)v34 + 2);
            v22 = v40;
            if ( v19[1] <= (int)v34 )
            {
              v30 = *v19;
              v38 = v34;
              v41 = *((_DWORD *)v20 + 10 * (int)v34 + 2);
              v31 = &v19[20 * v30 + 20 + *((unsigned int *)v19 + 3)];
              v32 = (__int64)&v31[*((unsigned __int8 *)v19 + 8)];
              v33 = sub_2F91020(v31, v32, &v41);
              v34 = v38;
              v22 = v40 | (v32 == (_QWORD)v33);
            }
            v43 = v21 | 0x100000000LL;
            v42 = a2 & 0xFFFFFFFFFFFFFFF9LL;
            v23 = a1 + 600;
            if ( v22 )
              goto LABEL_16;
LABEL_26:
            v39 = v34;
            v28 = sub_2FF8170(v23, *(_QWORD *)a2, a3, v18, v34);
            v34 = v39;
            HIDWORD(v43) = v28;
          }
          else
          {
            LODWORD(v43) = 3;
            v18 = 0;
            v23 = a1 + 600;
            v42 = a2 & 0xFFFFFFFFFFFFFFF9LL | 6;
            if ( !v40 )
              goto LABEL_26;
LABEL_16:
            HIDWORD(v43) = 0;
          }
          v24 = *(void (**)())(*(_QWORD *)v8 + 344LL);
          if ( v24 != nullsub_1667 )
            ((void (__fastcall *)(__int64, unsigned __int64, _QWORD, __int64, unsigned __int64))v24)(
              v8,
              a2,
              a3,
              v27,
              v34);
          sub_2F8F1B0(v27, (__int64)&v42, 1u, (__int64)v20, v34, (unsigned __int64)&v42);
        }
        else if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64))v20)(a1, a2, *v25) )
        {
          v25 = (__int64 *)(v26 + *(_QWORD *)(a1 + 1200));
          goto LABEL_24;
        }
        v15 = *(_QWORD *)(a1 + 1200);
        v25 = (__int64 *)(v15 + v26);
LABEL_21:
        result = *((unsigned int *)v25 + 5);
        if ( (_DWORD)result == -1 )
        {
          v11 = v36;
          break;
        }
        continue;
      }
    }
LABEL_30:
    v29 = *v35++;
    v11 += v29;
    if ( !(_WORD)v29 )
      return result;
  }
  return result;
}
