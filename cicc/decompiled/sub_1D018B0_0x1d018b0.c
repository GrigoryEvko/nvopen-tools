// Function: sub_1D018B0
// Address: 0x1d018b0
//
void __fastcall sub_1D018B0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // rax
  _QWORD *i; // r15
  unsigned __int64 v6; // rsi
  __int16 v7; // dx
  int v8; // ebx
  _DWORD *v9; // rdx
  unsigned int v10; // eax
  int v11; // [rsp-70h] [rbp-70h]
  __int16 v12; // [rsp-6Ch] [rbp-6Ch]
  unsigned int v13; // [rsp-60h] [rbp-60h] BYREF
  unsigned int v14; // [rsp-5Ch] [rbp-5Ch] BYREF
  __int64 v15; // [rsp-58h] [rbp-58h] BYREF
  __int64 v16; // [rsp-50h] [rbp-50h]

  if ( *(_BYTE *)(a1 + 44) && *(_QWORD *)a2 )
  {
    v3 = *(_QWORD **)(a2 + 32);
    v4 = 2LL * *(unsigned int *)(a2 + 40);
    for ( i = &v3[v4]; i != v3; *(_DWORD *)(*(_QWORD *)(a1 + 120) + 4LL * v13) += v14 )
    {
      while ( 1 )
      {
        if ( (*v3 & 6) == 0 )
        {
          v6 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
          v7 = *(_WORD *)(v6 + 224);
          if ( v7 )
          {
            *(_WORD *)(v6 + 224) = v7 - 1;
            v12 = v7 - 1;
            v11 = (unsigned __int16)(v7 - 1);
            sub_1D0E0B0(&v15, v6, *(_QWORD *)(a1 + 88));
            if ( v16 )
              break;
          }
        }
LABEL_5:
        v3 += 2;
        if ( i == v3 )
          goto LABEL_11;
      }
      if ( v12 )
      {
        do
        {
          sub_1D0DFF0(&v15);
          --v11;
          if ( !v16 )
            goto LABEL_5;
        }
        while ( v11 );
      }
      v3 += 2;
      sub_1D016F0(
        (__int64)&v15,
        *(_QWORD *)(a1 + 80),
        *(_QWORD *)(a1 + 64),
        *(_QWORD *)(a1 + 72),
        &v13,
        &v14,
        *(_QWORD *)(a1 + 56));
    }
LABEL_11:
    v8 = *(unsigned __int16 *)(a2 + 224);
    sub_1D0E0B0(&v15, a2, *(_QWORD *)(a1 + 88));
    while ( v16 )
    {
      if ( v8 <= 0 )
      {
        sub_1D016F0(
          (__int64)&v15,
          *(_QWORD *)(a1 + 80),
          *(_QWORD *)(a1 + 64),
          *(_QWORD *)(a1 + 72),
          &v13,
          &v14,
          *(_QWORD *)(a1 + 56));
        v9 = (_DWORD *)(*(_QWORD *)(a1 + 120) + 4LL * v13);
        v10 = *v9 - v14;
        if ( *v9 < v14 )
          v10 = 0;
        *v9 = v10;
      }
      --v8;
      sub_1D0DFF0(&v15);
    }
  }
}
