// Function: sub_34B4E10
// Address: 0x34b4e10
//
char __fastcall sub_34B4E10(_QWORD *a1, __int64 a2, int a3)
{
  int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r10
  __int64 v8; // r12
  unsigned int v9; // r14d
  __int64 v10; // r10
  __int64 v11; // rsi
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // rbx
  __int64 v17; // r12
  unsigned int v18; // r13d
  unsigned int v19; // edx
  bool v21; // [rsp+17h] [rbp-69h]
  __int64 v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+20h] [rbp-60h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+20h] [rbp-60h]
  int v27; // [rsp+2Ch] [rbp-54h]
  unsigned int v28; // [rsp+3Ch] [rbp-44h] BYREF
  __m128i v29; // [rsp+40h] [rbp-40h] BYREF

  v22 = a1[15];
  v4 = *(_DWORD *)(a2 + 44);
  if ( (v4 & 4) == 0 && (v4 & 8) != 0 )
  {
    LOBYTE(v5) = sub_2E88A90(a2, 128, 1);
    if ( (_BYTE)v5 )
      goto LABEL_4;
  }
  else
  {
    LOBYTE(v5) = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 7;
    if ( (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x80u) != 0LL )
    {
LABEL_4:
      v21 = 1;
      goto LABEL_5;
    }
  }
  v13 = *(_DWORD *)(a2 + 44);
  if ( (v13 & 4) != 0 || (v13 & 8) == 0 )
    v5 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 31) & 1LL;
  else
    LOBYTE(v5) = sub_2E88A90(a2, 0x80000000LL, 1);
  if ( (_BYTE)v5 )
    goto LABEL_4;
  v14 = a1[3];
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 920LL);
  if ( v15 != sub_2DB1B30 )
  {
    LOBYTE(v5) = ((__int64 (__fastcall *)(__int64, __int64))v15)(v14, a2);
    if ( (_BYTE)v5 )
      goto LABEL_4;
  }
  LODWORD(v5) = *(unsigned __int16 *)(a2 + 68) - 1;
  v21 = (unsigned int)v5 <= 1;
LABEL_5:
  if ( (*(_DWORD *)(a2 + 40) & 0xFFFFFF) != 0 )
  {
    v27 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
    v6 = 0;
    v7 = a2;
    do
    {
      LOBYTE(v5) = 5 * v6;
      v8 = *(_QWORD *)(v7 + 32) + 40 * v6;
      if ( !*(_BYTE *)v8 && (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
      {
        v9 = *(_DWORD *)(v8 + 8);
        if ( v9 )
        {
          v23 = v7;
          sub_34B45F0((__int64)a1, v9, a3);
          v10 = v23;
          if ( v21 )
          {
            sub_34B3410((_QWORD *)a1[15], v9, 0);
            v10 = v23;
          }
          v11 = *(_QWORD *)(v10 + 16);
          v12 = 0;
          if ( *(unsigned __int16 *)(v11 + 2) > (unsigned int)v6 )
          {
            v24 = v10;
            v12 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a1[3] + 16LL))(
                    a1[3],
                    v11,
                    (unsigned int)v6,
                    a1[4],
                    a1[1]);
            v10 = v24;
          }
          v29.m128i_i64[1] = v12;
          v25 = v10;
          v29.m128i_i64[0] = v8;
          v28 = v9;
          LOBYTE(v5) = sub_34B43E0((_QWORD *)(v22 + 56), &v28, &v29);
          v7 = v25;
        }
      }
      ++v6;
    }
    while ( v27 != (_DWORD)v6 );
    if ( *(_WORD *)(v7 + 68) == 7 )
    {
      v16 = *(_QWORD *)(v7 + 32);
      LODWORD(v5) = 5 * (*(_DWORD *)(v7 + 40) & 0xFFFFFF);
      v17 = v16 + 40LL * (*(_DWORD *)(v7 + 40) & 0xFFFFFF);
      if ( v16 != v17 )
      {
        v18 = 0;
        do
        {
          if ( !*(_BYTE *)v16 )
          {
            v19 = *(_DWORD *)(v16 + 8);
            if ( v19 )
            {
              if ( v18 )
                LOBYTE(v5) = sub_34B3410((_QWORD *)a1[15], v18, v19);
              else
                v18 = *(_DWORD *)(v16 + 8);
            }
          }
          v16 += 40;
        }
        while ( v17 != v16 );
      }
    }
  }
  return v5;
}
