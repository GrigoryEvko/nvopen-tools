// Function: sub_3351EF0
// Address: 0x3351ef0
//
__int64 __fastcall sub_3351EF0(_QWORD *a1, __int64 *a2, _DWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r12d
  _QWORD *v8; // rbx
  __int64 v9; // r14
  _WORD *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, unsigned __int16); // rax
  __int64 v14; // rcx
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  int v18; // r13d
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  char v23; // al
  __int64 v24; // rdi
  __int64 (__fastcall *v25)(__int64, unsigned __int16); // rax
  _QWORD *i; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  char v31[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v32; // [rsp+28h] [rbp-48h]
  unsigned __int16 v33; // [rsp+38h] [rbp-38h]

  v7 = 0;
  *a3 = 0;
  v8 = (_QWORD *)a2[5];
  v9 = 2LL * *((unsigned int *)a2 + 12);
  for ( i = &v8[v9]; i != v8; v8 += 2 )
  {
    if ( (*v8 & 6) == 0 )
    {
      v10 = (_WORD *)(*v8 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v10[125] )
      {
        sub_335E470(v31, v10, a1[11]);
        while ( v32 )
        {
          v12 = a1[10];
          v13 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v12 + 568LL);
          if ( v13 == sub_2FE3130 )
            v11 = *(_QWORD *)(v12 + 8LL * v33 + 3400);
          else
            v11 = v13(v12, v33);
          v7 -= (*(_DWORD *)(a1[15] + 4LL * *(unsigned __int16 *)(*(_QWORD *)v11 + 24LL)) < *(_DWORD *)(a1[18] + 4LL * *(unsigned __int16 *)(*(_QWORD *)v11 + 24LL)))
              - 1;
          sub_335E3B0(v31);
        }
      }
      else if ( *(int *)(*(_QWORD *)v10 + 24LL) < 0 )
      {
        ++*a3;
      }
    }
  }
  v14 = *a2;
  if ( *a2 )
  {
    v15 = *(_DWORD *)(v14 + 24);
    if ( v15 < 0 )
    {
      if ( *((_DWORD *)a2 + 53) )
      {
        v16 = 40LL * (unsigned int)~v15;
        v17 = *(_QWORD *)(a1[8] + 8LL) - v16;
        v18 = *(unsigned __int8 *)(v17 + 4);
        if ( *(_BYTE *)(v17 + 4) )
        {
          v19 = 0;
          do
          {
            v30 = v14;
            v22 = *(unsigned __int16 *)(*(_QWORD *)(v14 + 48) + 16 * v19);
            v23 = sub_33CF8A0(v14, (unsigned int)v19, v16, v14, a5, a6);
            v14 = v30;
            if ( v23 )
            {
              v24 = a1[10];
              v25 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v24 + 568LL);
              if ( v25 == sub_2FE3130 )
              {
                v20 = *(_QWORD *)(v24 + 8 * v22 + 3400);
              }
              else
              {
                v20 = v25(v24, v22);
                v14 = v30;
              }
              v21 = *(unsigned __int16 *)(*(_QWORD *)v20 + 24LL);
              v16 = *(unsigned int *)(a1[18] + 4 * v21);
              v7 = (*(_DWORD *)(a1[15] + 4 * v21) < (unsigned int)v16) + v7 - 1;
            }
            ++v19;
          }
          while ( v18 != (_DWORD)v19 );
        }
      }
    }
  }
  return v7;
}
