// Function: sub_FEDC10
// Address: 0xfedc10
//
void __fastcall sub_FEDC10(__int64 a1)
{
  __int64 v1; // rdx
  unsigned int v3; // ebx
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rcx
  __int64 *v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 *v12; // r15
  __int64 v13; // rax
  _DWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rax
  _QWORD *v17; // r14
  __int64 v18; // [rsp+8h] [rbp-48h]
  _DWORD v19[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v1 = 0;
  v3 = 0;
  v4 = *(_QWORD *)a1;
  *(_DWORD *)(a1 + 8) = 0;
  v5 = *(_QWORD *)(v4 + 64);
  if ( *(_QWORD *)(v4 + 72) != v5 )
  {
    do
    {
      v6 = v5 + 24 * v1;
      v7 = *(__int64 **)(v6 + 8);
      if ( v7 && *((_BYTE *)v7 + 8) )
      {
        do
        {
          v8 = v7;
          v7 = (__int64 *)*v7;
        }
        while ( v7 && *((_BYTE *)v7 + 8) );
        if ( *(_DWORD *)v6 != *(_DWORD *)v8[12] )
          goto LABEL_8;
      }
      v19[0] = v3;
      v9 = *(_QWORD *)(a1 + 32);
      if ( v9 == *(_QWORD *)(a1 + 40) )
      {
        sub_FEDA40((__int64 *)(a1 + 24), *(_QWORD *)(a1 + 32), v19);
        v10 = v19[0];
        v4 = *(_QWORD *)a1;
      }
      else
      {
        v10 = v3;
        if ( v9 )
        {
          *(_DWORD *)v9 = v3;
          *(_DWORD *)(v9 + 4) = 0;
          *(_QWORD *)(v9 + 8) = 0;
          *(_QWORD *)(v9 + 16) = 0;
          *(_QWORD *)(v9 + 24) = 0;
          *(_QWORD *)(v9 + 32) = 0;
          *(_QWORD *)(v9 + 40) = 0;
          *(_QWORD *)(v9 + 48) = 0;
          *(_QWORD *)(v9 + 56) = 0;
          *(_QWORD *)(v9 + 64) = 0;
          *(_QWORD *)(v9 + 72) = 0;
          *(_QWORD *)(v9 + 80) = 0;
          sub_FE91F0((__int64 *)(v9 + 8), 0);
          v9 = *(_QWORD *)(a1 + 32);
          v10 = v19[0];
          v4 = *(_QWORD *)a1;
        }
        *(_QWORD *)(a1 + 32) = v9 + 88;
      }
      v11 = *(_QWORD *)(v4 + 64) + 24 * v10;
      v12 = *(__int64 **)(v11 + 8);
      if ( !v12 )
        goto LABEL_22;
      v13 = *((unsigned int *)v12 + 3);
      v14 = (_DWORD *)v12[12];
      if ( (unsigned int)v13 > 1 )
      {
        if ( !sub_FDC990(v14, &v14[v13], (_DWORD *)v11) )
          goto LABEL_22;
      }
      else if ( *(_DWORD *)v11 != *v14 )
      {
        goto LABEL_22;
      }
      if ( *((_BYTE *)v12 + 8) )
      {
        v15 = *v12;
        if ( *v12
          && (v16 = *(unsigned int *)(v15 + 12), (unsigned int)v16 > 1)
          && (v18 = *v12, sub_FDC990(*(_DWORD **)(v15 + 96), (_DWORD *)(*(_QWORD *)(v15 + 96) + 4 * v16), (_DWORD *)v11)) )
        {
          v17 = (_QWORD *)(v18 + 152);
          if ( !*(_BYTE *)(v18 + 8) )
            v17 = v12 + 19;
        }
        else
        {
          v17 = v12 + 19;
        }
        goto LABEL_23;
      }
LABEL_22:
      v17 = (_QWORD *)(v11 + 16);
LABEL_23:
      *v17 = 0;
      v4 = *(_QWORD *)a1;
LABEL_8:
      v5 = *(_QWORD *)(v4 + 64);
      v1 = ++v3;
    }
    while ( v3 < 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(v4 + 72) - v5) >> 3) );
  }
  sub_FEAF00(a1);
}
