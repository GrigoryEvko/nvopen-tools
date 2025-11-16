// Function: sub_1D959A0
// Address: 0x1d959a0
//
__int64 __fastcall sub_1D959A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r13
  char v7; // r9
  __int64 v8; // rbx
  __int64 v12; // r12
  const void *v13; // r12
  size_t v14; // r14
  unsigned __int64 v15; // rbx
  __int64 v16; // rax
  __int64 result; // rax
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  __int64 v20; // rax
  __int64 i; // r10
  unsigned int v22; // ecx
  _DWORD *v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rsi
  char v28; // al
  char v29; // [rsp+7h] [rbp-59h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  char v33[49]; // [rsp+2Fh] [rbp-31h] BYREF

  v5 = a4;
  v6 = a2;
  v7 = a5 != 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 32LL);
  v32 = a1 + 576;
  v31 = a5 + 40;
  if ( a3 != v8 )
  {
    v12 = a5;
    do
    {
      while ( 1 )
      {
        if ( (unsigned __int16)(**(_WORD **)(v8 + 16) - 12) > 1u )
        {
          v18 = *(_QWORD *)(a1 + 544);
          v19 = *(__int64 (**)())(*(_QWORD *)v18 + 656LL);
          if ( v19 == sub_1D918C0
            || (v29 = v7, v28 = ((__int64 (__fastcall *)(__int64, __int64))v19)(v18, v8), v7 = v29, !v28) )
          {
            if ( v7 && (v33[0] = 1, (v7 = sub_1E17B50(v8, 0, v33)) != 0) )
            {
              v20 = *(_QWORD *)(v8 + 32);
              for ( i = v20 + 40LL * *(unsigned int *)(v8 + 40); i != v20; v20 += 40 )
              {
                if ( !*(_BYTE *)v20 )
                {
                  v22 = *(_DWORD *)(v20 + 8);
                  if ( v22 )
                  {
                    if ( (*(_BYTE *)(v20 + 3) & 0x10) != 0 )
                    {
                      if ( *(_QWORD *)(v12 + 72) )
                      {
                        v25 = *(_QWORD *)(v12 + 48);
                        if ( !v25 )
                          goto LABEL_18;
                        a5 = v31;
                        do
                        {
                          while ( 1 )
                          {
                            v26 = *(_QWORD *)(v25 + 16);
                            v27 = *(_QWORD *)(v25 + 24);
                            if ( v22 <= *(_DWORD *)(v25 + 32) )
                              break;
                            v25 = *(_QWORD *)(v25 + 24);
                            if ( !v27 )
                              goto LABEL_37;
                          }
                          a5 = v25;
                          v25 = *(_QWORD *)(v25 + 16);
                        }
                        while ( v26 );
LABEL_37:
                        if ( a5 == v31 || v22 < *(_DWORD *)(a5 + 32) )
                          goto LABEL_18;
                      }
                      else
                      {
                        v23 = *(_DWORD **)v12;
                        v24 = *(_QWORD *)v12 + 4LL * *(unsigned int *)(v12 + 8);
                        if ( *(_QWORD *)v12 == v24 )
                          goto LABEL_18;
                        while ( v22 != *v23 )
                        {
                          if ( (_DWORD *)v24 == ++v23 )
                            goto LABEL_18;
                        }
                        if ( (_DWORD *)v24 == v23 )
                          goto LABEL_18;
                      }
                    }
                  }
                }
              }
            }
            else
            {
LABEL_18:
              (*(void (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 544) + 696LL))(
                *(_QWORD *)(a1 + 544),
                v8,
                *(_QWORD *)a4,
                *(unsigned int *)(a4 + 8));
              sub_1D954F0(v8, v32);
              v7 = 0;
            }
          }
        }
        if ( (*(_BYTE *)v8 & 4) == 0 )
          break;
        v8 = *(_QWORD *)(v8 + 8);
        if ( a3 == v8 )
          goto LABEL_7;
      }
      while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
        v8 = *(_QWORD *)(v8 + 8);
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( a3 != v8 );
LABEL_7:
    v5 = a4;
    v6 = a2;
  }
  v13 = *(const void **)v5;
  v14 = 40LL * *(unsigned int *)(v5 + 8);
  v15 = *(unsigned int *)(v5 + 8);
  v16 = *(unsigned int *)(v6 + 224);
  if ( v15 > (unsigned __int64)*(unsigned int *)(v6 + 228) - v16 )
  {
    sub_16CD150(v6 + 216, (const void *)(v6 + 232), v15 + v16, 40, a5, v7);
    v16 = *(unsigned int *)(v6 + 224);
  }
  if ( v14 )
  {
    memcpy((void *)(*(_QWORD *)(v6 + 216) + 40 * v16), v13, v14);
    v16 = *(unsigned int *)(v6 + 224);
  }
  result = v15 + v16;
  *(_BYTE *)v6 &= ~4u;
  *(_DWORD *)(v6 + 224) = result;
  *(_DWORD *)(v6 + 4) = 0;
  return result;
}
