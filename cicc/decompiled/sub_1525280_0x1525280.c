// Function: sub_1525280
// Address: 0x1525280
//
void __fastcall sub_1525280(_DWORD *a1, unsigned __int64 a2, int a3)
{
  int v5; // r8d
  unsigned __int64 v6; // rbx
  unsigned int v7; // eax
  unsigned __int64 v8; // r9
  unsigned int v9; // r10d
  int v10; // ecx
  unsigned int v11; // r14d
  int v12; // r13d
  unsigned int v13; // ecx
  int v14; // r13d
  __int64 v15; // rdi
  __int64 v16; // rdx
  unsigned int v17; // ecx
  int v18; // edx
  unsigned int v19; // [rsp+4h] [rbp-4Ch]
  unsigned __int64 v20; // [rsp+8h] [rbp-48h]
  unsigned int v21; // [rsp+10h] [rbp-40h]
  int v22; // [rsp+14h] [rbp-3Ch]

  if ( a2 == (unsigned int)a2 )
  {
    sub_1524E40(a1, a2, a3);
  }
  else
  {
    v5 = a3 - 1;
    v6 = a2;
    v7 = 1 << (a3 - 1);
    v8 = v7;
    if ( a2 >= v7 )
    {
      v9 = v7 - 1;
      do
      {
        while ( 1 )
        {
          v10 = a1[2];
          v11 = v7 | v6 & v9;
          v12 = v11 << v10;
          v13 = a3 + v10;
          v14 = a1[3] | v12;
          a1[3] = v14;
          if ( v13 > 0x1F )
            break;
          a1[2] = v13;
          v6 >>= v5;
          if ( v6 < v8 )
            goto LABEL_11;
        }
        v15 = *(_QWORD *)a1;
        v16 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
        if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)a1 + 12LL) - v16 <= 3 )
        {
          v19 = v7;
          v20 = v8;
          v21 = v9;
          v22 = v5;
          sub_16CD150(v15, v15 + 16, v16 + 4, 1);
          v7 = v19;
          v8 = v20;
          v9 = v21;
          v16 = *(unsigned int *)(v15 + 8);
          v5 = v22;
        }
        *(_DWORD *)(*(_QWORD *)v15 + v16) = v14;
        v17 = 0;
        *(_DWORD *)(v15 + 8) += 4;
        v18 = a1[2];
        if ( v18 )
          v17 = v11 >> (32 - v18);
        a1[3] = v17;
        v6 >>= v5;
        a1[2] = ((_BYTE)a3 + (_BYTE)v18) & 0x1F;
      }
      while ( v6 >= v8 );
LABEL_11:
      LODWORD(a2) = v6;
    }
    sub_1524D80(a1, a2, a3);
  }
}
