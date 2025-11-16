// Function: sub_371BF00
// Address: 0x371bf00
//
void __fastcall sub_371BF00(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // r8
  unsigned int v5; // ebx
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v8; // r9
  __int64 v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-38h]

  v4 = a2 + 1;
  v5 = *(_DWORD *)(a1 + 56);
  if ( (unsigned int)v4 <= v5 )
  {
    v6 = *(_QWORD *)(a1 + 48);
    goto LABEL_3;
  }
  v7 = v5;
  v8 = (unsigned int)v4;
  v9 = v5;
  if ( (unsigned int)v4 != (unsigned __int64)v5 )
  {
    if ( (unsigned int)v4 >= (unsigned __int64)v5 )
    {
      if ( (unsigned int)v4 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
      {
        v13 = (unsigned int)v4;
        sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), (unsigned int)v4, 8u, v4, (unsigned int)v4);
        v9 = *(unsigned int *)(a1 + 56);
        LODWORD(v4) = a2 + 1;
        v8 = v13;
      }
      v6 = *(_QWORD *)(a1 + 48);
      v10 = (_QWORD *)(v6 + 8 * v9);
      v11 = (_QWORD *)(v6 + 8 * v8);
      if ( v10 != v11 )
      {
        do
        {
          if ( v10 )
            *v10 = 0;
          ++v10;
        }
        while ( v11 != v10 );
        v6 = *(_QWORD *)(a1 + 48);
      }
      *(_DWORD *)(a1 + 56) = v4;
      goto LABEL_14;
    }
    *(_DWORD *)(a1 + 56) = v4;
  }
  v6 = *(_QWORD *)(a1 + 48);
LABEL_14:
  v12 = v5 + 1;
  if ( (unsigned int)v4 > v5 + 1 )
  {
    while ( 1 )
    {
      *(_QWORD *)(v6 + 8 * v7) = 0;
      v6 = *(_QWORD *)(a1 + 48);
      if ( a2 == v12 )
        break;
      v7 = v12++;
    }
  }
LABEL_3:
  *(_QWORD *)(v6 + 8LL * a2) = a3;
}
