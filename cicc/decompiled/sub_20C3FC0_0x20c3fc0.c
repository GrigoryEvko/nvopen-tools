// Function: sub_20C3FC0
// Address: 0x20c3fc0
//
char __fastcall sub_20C3FC0(__int64 a1, __int64 a2, int a3)
{
  __int16 v5; // ax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned int v9; // r14d
  unsigned int v10; // r10d
  __int64 v11; // rsi
  __int64 v12; // rax
  __int16 v13; // ax
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // rbx
  unsigned int v17; // r14d
  __int64 v18; // r12
  unsigned int v19; // edx
  bool v21; // [rsp+17h] [rbp-69h]
  __int64 v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+20h] [rbp-60h]
  unsigned int v25; // [rsp+30h] [rbp-50h] BYREF
  __int64 v26; // [rsp+38h] [rbp-48h]
  __int64 v27; // [rsp+40h] [rbp-40h]

  v22 = *(_QWORD *)(a1 + 72);
  v5 = *(_WORD *)(a2 + 46);
  if ( (v5 & 4) == 0 && (v5 & 8) != 0 )
  {
    LOBYTE(v6) = sub_1E15D00(a2, 0x10u, 1);
    if ( (_BYTE)v6 )
      goto LABEL_4;
  }
  else
  {
    v6 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 4) & 1LL;
    if ( (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) & 0x10LL) != 0 )
    {
LABEL_4:
      v21 = 1;
      goto LABEL_5;
    }
  }
  v13 = *(_WORD *)(a2 + 46);
  if ( (v13 & 4) != 0 || (v13 & 8) == 0 )
    v6 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 27) & 1LL;
  else
    LOBYTE(v6) = sub_1E15D00(a2, 0x8000000u, 1);
  if ( (_BYTE)v6 )
    goto LABEL_4;
  v14 = *(_QWORD *)(a1 + 24);
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 656LL);
  if ( v15 != sub_1D918C0 )
  {
    LOBYTE(v6) = ((__int64 (__fastcall *)(__int64, __int64))v15)(v14, a2);
    if ( (_BYTE)v6 )
      goto LABEL_4;
  }
  v6 = *(_QWORD *)(a2 + 16);
  v21 = *(_WORD *)v6 == 1;
LABEL_5:
  if ( *(_DWORD *)(a2 + 40) )
  {
    v23 = *(unsigned int *)(a2 + 40);
    v7 = 0;
    do
    {
      v8 = *(_QWORD *)(a2 + 32) + 40 * v7;
      if ( !*(_BYTE *)v8 && (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
      {
        v9 = *(_DWORD *)(v8 + 8);
        if ( v9 )
        {
          sub_20C35E0(a1, v9, a3);
          v10 = v7;
          if ( v21 )
          {
            sub_20C2470(*(_QWORD **)(a1 + 72), v9, 0);
            v10 = v7;
          }
          v11 = *(_QWORD *)(a2 + 16);
          v12 = 0;
          if ( *(unsigned __int16 *)(v11 + 2) > v10 )
            v12 = sub_1F3AD60(*(_QWORD *)(a1 + 24), v11, v10, *(_QWORD **)(a1 + 32), *(_QWORD *)(a1 + 8));
          v27 = v12;
          v25 = v9;
          v26 = v8;
          sub_20C33D0(v22 + 56, (int *)&v25);
        }
      }
      ++v7;
    }
    while ( v23 != v7 );
    v6 = *(_QWORD *)(a2 + 16);
    if ( *(_WORD *)v6 == 6 )
    {
      v6 = *(unsigned int *)(a2 + 40);
      if ( (_DWORD)v6 )
      {
        v16 = 0;
        v17 = 0;
        v18 = 40 * v6;
        do
        {
          v6 = v16 + *(_QWORD *)(a2 + 32);
          if ( !*(_BYTE *)v6 )
          {
            v19 = *(_DWORD *)(v6 + 8);
            if ( v19 )
            {
              if ( v17 )
                LOBYTE(v6) = sub_20C2470(*(_QWORD **)(a1 + 72), v17, v19);
              else
                v17 = *(_DWORD *)(v6 + 8);
            }
          }
          v16 += 40;
        }
        while ( v16 != v18 );
      }
    }
  }
  return v6;
}
