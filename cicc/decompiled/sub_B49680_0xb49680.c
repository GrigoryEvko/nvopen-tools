// Function: sub_B49680
// Address: 0xb49680
//
__int64 __fastcall sub_B49680(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 *v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 *v9; // rcx
  __int64 v10; // r11
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r8
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // r14
  unsigned int v22; // r12d
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v29; // [rsp+8h] [rbp-38h]
  __int64 v30; // [rsp+8h] [rbp-38h]

  v5 = (__int64 *)a2;
  v6 = a1 + 32 * (a4 - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v7 = a2 + 56 * a3;
  if ( a2 != v7 )
  {
    v8 = a2;
    do
    {
      v9 = *(__int64 **)(v8 + 32);
      v10 = *(_QWORD *)(v8 + 40) - (_QWORD)v9;
      if ( v10 > 0 )
      {
        v11 = v6;
        a2 = (__int64)(*(_QWORD *)(v8 + 40) - (_QWORD)v9) >> 3;
        do
        {
          v12 = *v9;
          if ( *(_QWORD *)v11 )
          {
            v13 = *(_QWORD *)(v11 + 8);
            **(_QWORD **)(v11 + 16) = v13;
            if ( v13 )
              *(_QWORD *)(v13 + 16) = *(_QWORD *)(v11 + 16);
          }
          *(_QWORD *)v11 = v12;
          if ( v12 )
          {
            v14 = *(_QWORD *)(v12 + 16);
            *(_QWORD *)(v11 + 8) = v14;
            if ( v14 )
              *(_QWORD *)(v14 + 16) = v11 + 8;
            *(_QWORD *)(v11 + 16) = v12 + 16;
            *(_QWORD *)(v12 + 16) = v11;
          }
          ++v9;
          v11 += 32;
          --a2;
        }
        while ( a2 );
        v6 += 32 * (v10 >> 3);
      }
      v8 += 56;
    }
    while ( v8 != v7 );
  }
  v15 = a1;
  v16 = *(_QWORD *)sub_BD5C60(a1, a2);
  if ( *(char *)(a1 + 7) < 0 )
  {
    v17 = sub_BD2BC0(a1);
    v29 = v18 + v17;
    if ( *(char *)(v15 + 7) >= 0 )
      v19 = 0;
    else
      v19 = sub_BD2BC0(v15);
    v20 = v29;
    if ( v29 != v19 )
    {
      v30 = v6;
      v21 = v19;
      v22 = a4;
      v23 = v20;
      do
      {
        v24 = v5[1];
        v25 = *v5;
        v21 += 16;
        v5 += 7;
        v26 = sub_B71A20(v16, v25, v24);
        *(_DWORD *)(v21 - 8) = v22;
        *(_QWORD *)(v21 - 16) = v26;
        v22 += (*(v5 - 2) - *(v5 - 3)) >> 3;
        *(_DWORD *)(v21 - 4) = v22;
      }
      while ( v23 != v21 );
      return v30;
    }
  }
  return v6;
}
