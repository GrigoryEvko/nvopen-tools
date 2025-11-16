// Function: sub_1D01500
// Address: 0x1d01500
//
__int64 __fastcall sub_1D01500(_QWORD *a1, __int64 *a2, _DWORD *a3)
{
  unsigned int v4; // r12d
  _QWORD *v5; // rbx
  __int64 v6; // r14
  _WORD *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 (__fastcall *v10)(__int64, unsigned __int8); // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  int v13; // r13d
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r14
  char v17; // al
  __int64 v18; // rdi
  __int64 (__fastcall *v19)(__int64, unsigned __int8); // rax
  _QWORD *i; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+18h] [rbp-58h]
  char v25[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h]
  unsigned __int8 v27; // [rsp+38h] [rbp-38h]

  v4 = 0;
  *a3 = 0;
  v5 = (_QWORD *)a2[4];
  v6 = 2LL * *((unsigned int *)a2 + 10);
  for ( i = &v5[v6]; i != v5; v5 += 2 )
  {
    if ( (*v5 & 6) == 0 )
    {
      v7 = (_WORD *)(*v5 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v7[112] )
      {
        sub_1D0E0B0(v25, v7, a1[11]);
        while ( v26 )
        {
          v9 = a1[10];
          v10 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v9 + 296LL);
          if ( v10 == sub_1D00B40 )
            v8 = *(_QWORD *)(v9 + 8LL * v27 + 1272);
          else
            v8 = v10(v9, v27);
          v4 -= (*(_DWORD *)(a1[15] + 4LL * *(unsigned __int16 *)(*(_QWORD *)v8 + 24LL)) < *(_DWORD *)(a1[18] + 4LL * *(unsigned __int16 *)(*(_QWORD *)v8 + 24LL)))
              - 1;
          sub_1D0DFF0(v25);
        }
      }
      else if ( *(__int16 *)(*(_QWORD *)v7 + 24LL) < 0 )
      {
        ++*a3;
      }
    }
  }
  v11 = *a2;
  if ( *a2 )
  {
    if ( *(__int16 *)(v11 + 24) < 0 )
    {
      if ( *((_DWORD *)a2 + 51) )
      {
        v12 = *(_QWORD *)(a1[8] + 8LL) + ((__int64)~*(__int16 *)(v11 + 24) << 6);
        v13 = *(unsigned __int8 *)(v12 + 4);
        if ( *(_BYTE *)(v12 + 4) )
        {
          v14 = 0;
          do
          {
            v24 = v11;
            v16 = *(unsigned __int8 *)(*(_QWORD *)(v11 + 40) + 16 * v14);
            v17 = sub_1D18C40(v11);
            v11 = v24;
            if ( v17 )
            {
              v18 = a1[10];
              v19 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v18 + 296LL);
              if ( v19 == sub_1D00B40 )
              {
                v15 = *(_QWORD *)(v18 + 8 * v16 + 1272);
              }
              else
              {
                v15 = v19(v18, v16);
                v11 = v24;
              }
              v4 = (*(_DWORD *)(a1[15] + 4LL * *(unsigned __int16 *)(*(_QWORD *)v15 + 24LL)) < *(_DWORD *)(a1[18] + 4LL * *(unsigned __int16 *)(*(_QWORD *)v15 + 24LL)))
                 + v4
                 - 1;
            }
            ++v14;
          }
          while ( v13 != (_DWORD)v14 );
        }
      }
    }
  }
  return v4;
}
