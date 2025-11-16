// Function: sub_2BDEC80
// Address: 0x2bdec80
//
void *__fastcall sub_2BDEC80(__int64 a1, __int64 a2)
{
  __int64 *v4; // rdi
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // rax
  unsigned int v8; // r14d
  __int64 v10; // rcx
  _QWORD *v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rsi
  _QWORD *v14; // rsi
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdi
  int v19; // eax

  v4 = (__int64 *)(a1 + 40);
  *(v4 - 5) = *(_QWORD *)a2;
  *(v4 - 4) = *(_QWORD *)(a2 + 8);
  *(v4 - 3) = *(_QWORD *)(a2 + 16);
  *(v4 - 2) = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a2 + 24) = 0;
  *(v4 - 1) = (__int64)&unk_4A23850;
  *(_QWORD *)(a1 + 40) = a1 + 56;
  sub_2BDC2F0(v4, *(_BYTE **)(a2 + 40), *(_QWORD *)(a2 + 40) + *(_QWORD *)(a2 + 48));
  *(_QWORD *)(a1 + 80) = 0x600000000LL;
  *(_QWORD *)(a1 + 32) = &unk_4A23878;
  v7 = (_QWORD *)(a1 + 88);
  *(_QWORD *)(a1 + 72) = a1 + 88;
  v8 = *(_DWORD *)(a2 + 80);
  if ( v8 && a1 + 72 != a2 + 72 )
  {
    v10 = *(_QWORD *)(a2 + 72);
    v11 = (_QWORD *)(a2 + 88);
    if ( v10 == a2 + 88 )
    {
      v12 = v8;
      if ( v8 > 6 )
      {
        sub_2BDE4F0(a1 + 72, v8, (__int64)v11, v10, v5, v6);
        v7 = *(_QWORD **)(a1 + 72);
        v11 = *(_QWORD **)(a2 + 72);
        v12 = *(unsigned int *)(a2 + 80);
      }
      v13 = v12;
      if ( v13 * 8 )
      {
        v14 = &v7[v13];
        do
        {
          if ( v7 )
          {
            *v7 = *v11;
            *v11 = 0;
          }
          ++v7;
          ++v11;
        }
        while ( v7 != v14 );
        v15 = *(_QWORD *)(a2 + 72);
        v16 = *(unsigned int *)(a2 + 80);
        *(_DWORD *)(a1 + 80) = v8;
        v17 = v15 + 8 * v16;
        while ( v17 != v15 )
        {
          while ( 1 )
          {
            v18 = *(_QWORD *)(v17 - 8);
            v17 -= 8;
            if ( !v18 )
              break;
            (*(void (__fastcall **)(__int64, _QWORD *, _QWORD *))(*(_QWORD *)v18 + 8LL))(v18, v14, v11);
            if ( v17 == v15 )
              goto LABEL_16;
          }
        }
      }
      else
      {
        *(_DWORD *)(a1 + 80) = v8;
      }
LABEL_16:
      *(_DWORD *)(a2 + 80) = 0;
    }
    else
    {
      v19 = *(_DWORD *)(a2 + 84);
      *(_QWORD *)(a1 + 72) = v10;
      *(_DWORD *)(a1 + 80) = v8;
      *(_DWORD *)(a1 + 84) = v19;
      *(_QWORD *)(a2 + 72) = v11;
      *(_QWORD *)(a2 + 80) = 0;
    }
  }
  *(_QWORD *)(a1 + 32) = &unk_4A34660;
  return &unk_4A34660;
}
