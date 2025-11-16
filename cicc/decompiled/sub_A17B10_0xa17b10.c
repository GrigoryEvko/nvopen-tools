// Function: sub_A17B10
// Address: 0xa17b10
//
void __fastcall sub_A17B10(__int64 a1, unsigned int a2, int a3)
{
  int v3; // ecx
  int v4; // r13d
  unsigned int v5; // ecx
  int v6; // r13d
  _QWORD *v7; // r14
  char v8; // r12
  __int64 v9; // rax
  unsigned int v10; // eax
  int v11; // edx

  v3 = *(_DWORD *)(a1 + 48);
  v4 = a2 << v3;
  v5 = a3 + v3;
  v6 = *(_DWORD *)(a1 + 52) | v4;
  *(_DWORD *)(a1 + 52) = v6;
  if ( v5 > 0x1F )
  {
    v7 = *(_QWORD **)(a1 + 24);
    v8 = a3;
    v9 = v7[1];
    if ( (unsigned __int64)(v9 + 4) > v7[2] )
    {
      sub_C8D290(*(_QWORD *)(a1 + 24), v7 + 3, v9 + 4, 1);
      v9 = v7[1];
    }
    *(_DWORD *)(*v7 + v9) = v6;
    v10 = 0;
    v7[1] += 4LL;
    v11 = *(_DWORD *)(a1 + 48);
    if ( v11 )
      v10 = a2 >> (32 - v11);
    *(_DWORD *)(a1 + 52) = v10;
    *(_DWORD *)(a1 + 48) = (v8 + (_BYTE)v11) & 0x1F;
  }
  else
  {
    *(_DWORD *)(a1 + 48) = v5;
  }
}
