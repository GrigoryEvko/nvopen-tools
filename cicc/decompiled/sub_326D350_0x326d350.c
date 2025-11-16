// Function: sub_326D350
// Address: 0x326d350
//
void __fastcall sub_326D350(__int64 a1, _QWORD *a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  __int64 v5; // r8
  __int16 *v8; // rdi
  __int16 v9; // ax
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int16 v12; // dx
  unsigned __int16 v13; // r12
  __int64 v14; // rax
  unsigned __int64 v15; // r13
  __int64 v16; // rdx
  char v17; // r14
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  const __m128i *v22; // rcx
  bool v23; // al
  unsigned int v24; // [rsp-80h] [rbp-80h]
  unsigned __int16 v25; // [rsp-68h] [rbp-68h] BYREF
  __int64 v26; // [rsp-60h] [rbp-60h]
  unsigned __int16 v27; // [rsp-58h] [rbp-58h] BYREF
  __int64 v28; // [rsp-50h] [rbp-50h]
  __int64 v29; // [rsp-48h] [rbp-48h]
  __int64 v30; // [rsp-40h] [rbp-40h]
  __int64 v31; // [rsp-38h] [rbp-38h]
  __int64 v32; // [rsp-30h] [rbp-30h]

  v5 = *a2 + 16LL * a5;
  if ( !*(_QWORD *)v5 )
  {
    *(_QWORD *)v5 = a3;
    *(_DWORD *)(v5 + 8) = a4;
    v8 = *(__int16 **)a1;
    v9 = *v8;
    if ( *v8 )
    {
      if ( (unsigned __int16)(v9 - 2) <= 7u
        || (unsigned __int16)(v9 - 17) <= 0x6Cu
        || (unsigned __int16)(v9 - 176) <= 0x1Fu )
      {
LABEL_6:
        v10 = *(_QWORD *)(a1 + 8);
        v11 = *(_QWORD *)(a3 + 48) + 16LL * a4;
        v12 = *(_WORD *)v11;
        v13 = *(_WORD *)v10;
        v14 = *(_QWORD *)(v11 + 8);
        v25 = v12;
        v26 = v14;
        if ( v12 == v13 )
        {
          v22 = (const __m128i *)v10;
          if ( v12 || v14 == *(_QWORD *)(v10 + 8) )
          {
LABEL_18:
            *(__m128i *)v10 = _mm_loadu_si128(v22);
            return;
          }
          v28 = v14;
          v27 = 0;
        }
        else
        {
          v27 = v12;
          v28 = v14;
          if ( v12 )
          {
            if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
              goto LABEL_28;
            v15 = *(_QWORD *)&byte_444C4A0[16 * v12 - 16];
            v17 = byte_444C4A0[16 * v12 - 8];
LABEL_9:
            if ( !v13 )
            {
              v18 = sub_3007260(v10);
              v20 = v19;
              v29 = v18;
              v21 = v18;
              v30 = v20;
              goto LABEL_11;
            }
            if ( v13 != 1 && (unsigned __int16)(v13 - 504) > 7u )
            {
              v21 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
              LOBYTE(v20) = byte_444C4A0[16 * v13 - 8];
LABEL_11:
              if ( ((_BYTE)v20 || !v17) && v21 >= v15 )
                v22 = (const __m128i *)v10;
              else
                v22 = (const __m128i *)&v25;
              goto LABEL_18;
            }
LABEL_28:
            BUG();
          }
        }
        v31 = sub_3007260((__int64)&v27);
        v15 = v31;
        v32 = v16;
        v17 = v16;
        goto LABEL_9;
      }
    }
    else
    {
      v24 = a4;
      v23 = sub_3007070((__int64)v8);
      a4 = v24;
      if ( v23 )
        goto LABEL_6;
    }
  }
}
