// Function: sub_371C0E0
// Address: 0x371c0e0
//
void __fastcall sub_371C0E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  int v8; // eax
  __int64 v9; // rsi
  int v10; // edx
  unsigned int v11; // eax
  __int64 *v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rsi
  _QWORD *v17; // rax
  int v18; // r8d
  int v19; // ecx
  int v20; // r8d
  __int64 v21[3]; // [rsp+8h] [rbp-18h] BYREF

  v6 = a2;
  v21[0] = a2;
  sub_371B8E0(a1 + 128, a2, a3, a4, a5, a6);
  v8 = *(_DWORD *)(a1 + 24);
  v9 = *(_QWORD *)(a1 + 8);
  if ( v8 )
  {
    v10 = v8 - 1;
    v11 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v12 = (__int64 *)(v9 + 8LL * v11);
    v13 = *v12;
    if ( v6 == *v12 )
    {
LABEL_3:
      *v12 = -8192;
      v14 = *(unsigned int *)(a1 + 40);
      --*(_DWORD *)(a1 + 16);
      v15 = *(_QWORD **)(a1 + 32);
      ++*(_DWORD *)(a1 + 20);
      v16 = (__int64)&v15[v14];
      v17 = sub_371B710(v15, v16, v21);
      if ( v17 + 1 != (_QWORD *)v16 )
      {
        memmove(v17, v17 + 1, v16 - (_QWORD)(v17 + 1));
        v18 = *(_DWORD *)(a1 + 40);
        v6 = v21[0];
      }
      *(_DWORD *)(a1 + 40) = v18 - 1;
    }
    else
    {
      v19 = 1;
      while ( v13 != -4096 )
      {
        v20 = v19 + 1;
        v11 = v10 & (v19 + v11);
        v12 = (__int64 *)(v9 + 8LL * v11);
        v13 = *v12;
        if ( v6 == *v12 )
          goto LABEL_3;
        v19 = v20;
      }
    }
  }
  sub_B9A090(*(_QWORD *)(v6 + 16), "sandboxvec", 0xAu, 0);
}
