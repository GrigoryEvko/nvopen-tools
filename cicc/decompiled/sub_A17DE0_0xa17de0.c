// Function: sub_A17DE0
// Address: 0xa17de0
//
void __fastcall sub_A17DE0(__int64 a1, unsigned __int64 a2, int a3)
{
  int v5; // r9d
  unsigned __int64 v6; // rbx
  unsigned int v7; // r8d
  unsigned __int64 v8; // r10
  int v9; // eax
  int v10; // ecx
  unsigned int v11; // r14d
  int v12; // r13d
  unsigned int v13; // ecx
  int v14; // r13d
  _QWORD *v15; // rdi
  __int64 v16; // rdx
  unsigned int v17; // ecx
  int v18; // edx
  unsigned __int64 v19; // [rsp+0h] [rbp-50h]
  unsigned int v20; // [rsp+Ch] [rbp-44h]
  int v21; // [rsp+10h] [rbp-40h]
  int v22; // [rsp+14h] [rbp-3Ch]

  if ( a2 == (unsigned int)a2 )
  {
    sub_A17CC0(a1, a2, a3);
  }
  else
  {
    v5 = a3 - 1;
    v6 = a2;
    v7 = 1 << (a3 - 1);
    v8 = v7;
    if ( a2 >= v7 )
    {
      v9 = ~(-1 << (a3 - 1));
      do
      {
        while ( 1 )
        {
          v10 = *(_DWORD *)(a1 + 48);
          v11 = v7 | v6 & v9;
          v12 = v11 << v10;
          v13 = a3 + v10;
          v14 = *(_DWORD *)(a1 + 52) | v12;
          *(_DWORD *)(a1 + 52) = v14;
          if ( v13 > 0x1F )
            break;
          *(_DWORD *)(a1 + 48) = v13;
          v6 >>= v5;
          if ( v6 < v8 )
            goto LABEL_11;
        }
        v15 = *(_QWORD **)(a1 + 24);
        v16 = v15[1];
        if ( (unsigned __int64)(v16 + 4) > v15[2] )
        {
          v19 = v8;
          v20 = v7;
          v21 = v9;
          v22 = v5;
          sub_C8D290(v15, v15 + 3, v16 + 4, 1);
          v8 = v19;
          v7 = v20;
          v9 = v21;
          v16 = v15[1];
          v5 = v22;
        }
        *(_DWORD *)(*v15 + v16) = v14;
        v17 = 0;
        v15[1] += 4LL;
        v18 = *(_DWORD *)(a1 + 48);
        if ( v18 )
          v17 = v11 >> (32 - v18);
        *(_DWORD *)(a1 + 52) = v17;
        v6 >>= v5;
        *(_DWORD *)(a1 + 48) = ((_BYTE)a3 + (_BYTE)v18) & 0x1F;
      }
      while ( v6 >= v8 );
LABEL_11:
      LODWORD(a2) = v6;
    }
    sub_A17B10(a1, a2, a3);
  }
}
