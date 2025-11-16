// Function: sub_B36280
// Address: 0xb36280
//
__int64 __fastcall sub_B36280(
        unsigned int **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v9; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rbx
  unsigned int v20; // r15d
  unsigned int *v21; // rdx
  unsigned int *v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-78h]
  char v29[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v30; // [rsp+40h] [rbp-40h]

  v9 = (*(__int64 (__fastcall **)(unsigned int *))(*(_QWORD *)a1[10] + 72LL))(a1[10]);
  if ( !v9 )
  {
    v30 = 257;
    v11 = sub_BD2C40(72, 3);
    v9 = v11;
    if ( v11 )
    {
      v26 = v11;
      sub_B44260(v11, *(_QWORD *)(a3 + 8), 57, 3, 0, 0);
      if ( *(_QWORD *)(v9 - 96) )
      {
        v12 = *(_QWORD *)(v9 - 88);
        **(_QWORD **)(v9 - 80) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v9 - 80);
      }
      *(_QWORD *)(v9 - 96) = a2;
      if ( a2 )
      {
        v13 = *(_QWORD *)(a2 + 16);
        *(_QWORD *)(v9 - 88) = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = v9 - 88;
        *(_QWORD *)(v9 - 80) = a2 + 16;
        *(_QWORD *)(a2 + 16) = v9 - 96;
      }
      if ( *(_QWORD *)(v9 - 64) )
      {
        v14 = *(_QWORD *)(v9 - 56);
        **(_QWORD **)(v9 - 48) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = *(_QWORD *)(v9 - 48);
      }
      *(_QWORD *)(v9 - 64) = a3;
      v15 = *(_QWORD *)(a3 + 16);
      *(_QWORD *)(v9 - 56) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = v9 - 56;
      *(_QWORD *)(v9 - 48) = a3 + 16;
      *(_QWORD *)(a3 + 16) = v9 - 64;
      if ( *(_QWORD *)(v9 - 32) )
      {
        v16 = *(_QWORD *)(v9 - 24);
        **(_QWORD **)(v9 - 16) = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = *(_QWORD *)(v9 - 16);
      }
      *(_QWORD *)(v9 - 32) = a4;
      if ( a4 )
      {
        v17 = *(_QWORD *)(a4 + 16);
        *(_QWORD *)(v9 - 24) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = v9 - 24;
        *(_QWORD *)(v9 - 16) = a4 + 16;
        *(_QWORD *)(a4 + 16) = v9 - 32;
      }
      sub_BD6B50(v9, v29);
    }
    else
    {
      v26 = 0;
    }
    if ( a7 && (*(_BYTE *)(a7 + 7) & 0x20) != 0 )
    {
      v18 = sub_B91C10(a7, 2);
      if ( (*(_BYTE *)(a7 + 7) & 0x20) != 0 )
      {
        v19 = sub_B91C10(a7, 15);
        if ( v18 )
          sub_B99FD0(v26, 2, v18);
        if ( v19 )
          sub_B99FD0(v26, 15, v19);
      }
      else if ( v18 )
      {
        sub_B99FD0(v26, 2, v18);
      }
    }
    if ( (unsigned __int8)sub_920620(v26) )
    {
      v20 = *((_DWORD *)a1 + 26);
      if ( BYTE4(a5) )
        v20 = a5;
      v21 = a1[12];
      if ( v21 )
        sub_B99FD0(v9, 3, v21);
      sub_B45150(v9, v20);
    }
    (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v9,
      a6,
      a1[7],
      a1[8]);
    v22 = *a1;
    v23 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    if ( *a1 != (unsigned int *)v23 )
    {
      do
      {
        v24 = *((_QWORD *)v22 + 1);
        v25 = *v22;
        v22 += 4;
        sub_B99FD0(v9, v25, v24);
      }
      while ( (unsigned int *)v23 != v22 );
    }
  }
  return v9;
}
