// Function: sub_3578190
// Address: 0x3578190
//
void __fastcall sub_3578190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 (*v8)(void); // rax
  __int64 v9; // r14
  __int64 v10; // rbx
  int v11; // eax
  int v12; // eax
  __int64 (*v13)(); // rax
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  _DWORD *v17; // rdx
  int v18; // eax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // eax
  __int64 *v22; // rdx
  __int64 v23; // rcx
  __int64 *v24; // rax
  __int64 v25; // rax
  __int64 v26; // [rsp+18h] [rbp-78h]
  _BYTE *v27; // [rsp+20h] [rbp-70h] BYREF
  __int64 v28; // [rsp+28h] [rbp-68h]
  _BYTE v29[96]; // [rsp+30h] [rbp-60h] BYREF

  v6 = 0;
  v7 = *(_QWORD *)(a1 + 216);
  v8 = *(__int64 (**)(void))(**(_QWORD **)(v7 + 16) + 128LL);
  if ( v8 != sub_2DAC790 )
  {
    v25 = v8();
    v7 = *(_QWORD *)(a1 + 216);
    v6 = v25;
  }
  v27 = v29;
  v28 = 0x600000000LL;
  v26 = *(_QWORD *)(v7 + 328);
  if ( v26 != v7 + 320 )
  {
    do
    {
      v9 = *(_QWORD *)(v26 + 56);
      v10 = v26 + 48;
      if ( v26 + 48 != v9 )
      {
        while ( 1 )
        {
          v11 = *(_DWORD *)(v9 + 44);
          if ( (v11 & 4) != 0 || (v11 & 8) == 0 )
          {
            if ( (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) & 0x80u) != 0LL )
              goto LABEL_10;
          }
          else if ( sub_2E88A90(v9, 128, 1) )
          {
            goto LABEL_10;
          }
          if ( (unsigned int)*(unsigned __int16 *)(v9 + 68) - 1 <= 1
            && (*(_BYTE *)(*(_QWORD *)(v9 + 32) + 64LL) & 0x10) != 0
            || ((v14 = *(_DWORD *)(v9 + 44), (v14 & 4) != 0) || (v14 & 8) == 0
              ? (v15 = (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) >> 20) & 1LL)
              : (LOBYTE(v15) = sub_2E88A90(v9, 0x100000, 1)),
                (_BYTE)v15) )
          {
            v16 = *(_QWORD *)(v9 + 48);
            v17 = (_DWORD *)(v16 & 0xFFFFFFFFFFFFFFF8LL);
            if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            {
              v18 = v16 & 7;
              if ( !v18 )
              {
                *(_QWORD *)(v9 + 48) = v17;
LABEL_10:
                *(_BYTE *)(a1 + 1640) = 1;
                goto LABEL_11;
              }
              if ( v18 == 3 && *v17 )
                goto LABEL_10;
            }
          }
LABEL_11:
          if ( (unsigned int)*(unsigned __int16 *)(v9 + 68) - 1 > 1
            || (*(_BYTE *)(*(_QWORD *)(v9 + 32) + 64LL) & 8) == 0 )
          {
            v12 = *(_DWORD *)(v9 + 44);
            if ( (v12 & 4) != 0 || (v12 & 8) == 0 )
            {
              if ( (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) & 0x80000LL) == 0 )
              {
LABEL_16:
                v13 = *(__int64 (**)())(*(_QWORD *)v6 + 1512LL);
                if ( v13 == sub_2FDC830 )
                  goto LABEL_17;
                goto LABEL_41;
              }
            }
            else if ( !sub_2E88A90(v9, 0x80000, 1) )
            {
              goto LABEL_16;
            }
          }
          v19 = (unsigned int)v28;
          v20 = (unsigned int)v28 + 1LL;
          if ( v20 > HIDWORD(v28) )
          {
            sub_C8D5F0((__int64)&v27, v29, v20, 8u, a5, a6);
            v19 = (unsigned int)v28;
          }
          *(_QWORD *)&v27[8 * v19] = v9;
          LODWORD(v28) = v28 + 1;
          v13 = *(__int64 (**)())(*(_QWORD *)v6 + 1512LL);
          if ( v13 == sub_2FDC830 )
            goto LABEL_17;
LABEL_41:
          v21 = ((__int64 (__fastcall *)(__int64, __int64))v13)(v6, v9);
          if ( v21 == 1 )
          {
            if ( !*(_BYTE *)(a1 + 1284) )
              goto LABEL_51;
            v24 = *(__int64 **)(a1 + 1264);
            v23 = *(unsigned int *)(a1 + 1276);
            v22 = &v24[v23];
            if ( v24 != v22 )
            {
              while ( v9 != *v24 )
              {
                if ( v22 == ++v24 )
                  goto LABEL_49;
              }
              goto LABEL_17;
            }
LABEL_49:
            if ( (unsigned int)v23 < *(_DWORD *)(a1 + 1272) )
            {
              *(_DWORD *)(a1 + 1276) = v23 + 1;
              *v22 = v9;
              ++*(_QWORD *)(a1 + 1256);
            }
            else
            {
LABEL_51:
              sub_C8CC70(a1 + 1256, v9, (__int64)v22, v23, a5, a6);
            }
          }
          else if ( v21 == 2 )
          {
            sub_3577FF0(a1, v9, (__int64)v22, v23, a5, a6);
          }
LABEL_17:
          if ( (*(_BYTE *)v9 & 4) != 0 )
          {
            v9 = *(_QWORD *)(v9 + 8);
            if ( v10 == v9 )
              break;
          }
          else
          {
            while ( (*(_BYTE *)(v9 + 44) & 8) != 0 )
              v9 = *(_QWORD *)(v9 + 8);
            v9 = *(_QWORD *)(v9 + 8);
            if ( v10 == v9 )
              break;
          }
        }
      }
      v26 = *(_QWORD *)(v26 + 8);
    }
    while ( v7 + 320 != v26 );
    v7 = *(_QWORD *)(a1 + 216);
  }
  sub_3574C70(a1, v7);
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
}
