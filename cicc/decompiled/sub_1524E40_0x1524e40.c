// Function: sub_1524E40
// Address: 0x1524e40
//
void __fastcall sub_1524E40(_DWORD *a1, unsigned int a2, int a3)
{
  int v3; // eax
  int v4; // r8d
  unsigned int v5; // r15d
  unsigned int v6; // r13d
  unsigned int v8; // r9d
  int v9; // ecx
  unsigned int v10; // r14d
  unsigned int v11; // ebx
  unsigned int v12; // ecx
  int v13; // ebx
  __int64 v14; // rdi
  __int64 v15; // rdx
  unsigned int v16; // ecx
  int v17; // edx
  int v18; // [rsp+Ch] [rbp-44h]
  unsigned int v19; // [rsp+10h] [rbp-40h]
  int v20; // [rsp+14h] [rbp-3Ch]

  v3 = a3 - 1;
  v4 = a3;
  v5 = 1 << (a3 - 1);
  v6 = a2;
  if ( v5 <= a2 )
  {
    v8 = v5 - 1;
    do
    {
      while ( 1 )
      {
        v9 = a1[2];
        v10 = v5 | v6 & v8;
        v11 = v10 << v9;
        v12 = v4 + v9;
        v13 = a1[3] | v11;
        a1[3] = v13;
        if ( v12 > 0x1F )
          break;
        a1[2] = v12;
        v6 >>= v3;
        if ( v5 > v6 )
          goto LABEL_10;
      }
      v14 = *(_QWORD *)a1;
      v15 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
      if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)a1 + 12LL) - v15 <= 3 )
      {
        v20 = v3;
        v18 = v4;
        v19 = v8;
        sub_16CD150(v14, v14 + 16, v15 + 4, 1);
        v4 = v18;
        v8 = v19;
        v3 = v20;
        v15 = *(unsigned int *)(v14 + 8);
      }
      *(_DWORD *)(*(_QWORD *)v14 + v15) = v13;
      v16 = 0;
      *(_DWORD *)(v14 + 8) += 4;
      v17 = a1[2];
      if ( v17 )
        v16 = v10 >> (32 - v17);
      a1[3] = v16;
      v6 >>= v3;
      a1[2] = ((_BYTE)v4 + (_BYTE)v17) & 0x1F;
    }
    while ( v5 <= v6 );
  }
LABEL_10:
  sub_1524D80(a1, v6, v4);
}
