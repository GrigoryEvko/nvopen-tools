// Function: sub_A17CC0
// Address: 0xa17cc0
//
void __fastcall sub_A17CC0(__int64 a1, unsigned int a2, int a3)
{
  int v3; // r9d
  int v4; // r8d
  unsigned int v5; // r15d
  unsigned int v6; // r13d
  int v8; // eax
  int v9; // ecx
  unsigned int v10; // r14d
  unsigned int v11; // ebx
  unsigned int v12; // ecx
  int v13; // ebx
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  unsigned int v16; // ecx
  int v17; // edx
  int v18; // [rsp+Ch] [rbp-44h]
  int v19; // [rsp+10h] [rbp-40h]
  int v20; // [rsp+14h] [rbp-3Ch]

  v3 = a3 - 1;
  v4 = a3;
  v5 = 1 << (a3 - 1);
  v6 = a2;
  if ( v5 <= a2 )
  {
    v8 = ~(-1 << (a3 - 1));
    do
    {
      while ( 1 )
      {
        v9 = *(_DWORD *)(a1 + 48);
        v10 = v5 | v6 & v8;
        v11 = v10 << v9;
        v12 = v4 + v9;
        v13 = *(_DWORD *)(a1 + 52) | v11;
        *(_DWORD *)(a1 + 52) = v13;
        if ( v12 > 0x1F )
          break;
        *(_DWORD *)(a1 + 48) = v12;
        v6 >>= v3;
        if ( v5 > v6 )
          goto LABEL_10;
      }
      v14 = *(_QWORD **)(a1 + 24);
      v15 = v14[1];
      if ( (unsigned __int64)(v15 + 4) > v14[2] )
      {
        v18 = v4;
        v19 = v8;
        v20 = v3;
        sub_C8D290(v14, v14 + 3, v15 + 4, 1);
        v4 = v18;
        v8 = v19;
        v3 = v20;
        v15 = v14[1];
      }
      *(_DWORD *)(*v14 + v15) = v13;
      v16 = 0;
      v14[1] += 4LL;
      v17 = *(_DWORD *)(a1 + 48);
      if ( v17 )
        v16 = v10 >> (32 - v17);
      *(_DWORD *)(a1 + 52) = v16;
      v6 >>= v3;
      *(_DWORD *)(a1 + 48) = ((_BYTE)v4 + (_BYTE)v17) & 0x1F;
    }
    while ( v5 <= v6 );
  }
LABEL_10:
  sub_A17B10(a1, v6, v4);
}
