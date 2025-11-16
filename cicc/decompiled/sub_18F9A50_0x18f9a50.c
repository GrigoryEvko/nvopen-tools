// Function: sub_18F9A50
// Address: 0x18f9a50
//
__int64 __fastcall sub_18F9A50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // rbx
  unsigned int v7; // r13d
  __int64 v8; // r12
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r8
  unsigned int v13; // esi
  __int64 *v14; // rdx
  __int64 v15; // r10
  int v18; // edx
  int v19; // ecx

  v5 = a1 + 72;
  if ( *(_QWORD *)(a1 + 80) == a1 + 72 )
  {
    return 0;
  }
  else
  {
    v7 = 0;
    v8 = *(_QWORD *)(a1 + 80);
    do
    {
      v10 = v8 - 24;
      if ( !v8 )
        v10 = 0;
      v11 = *(unsigned int *)(a4 + 48);
      if ( (_DWORD)v11 )
      {
        v12 = *(_QWORD *)(a4 + 32);
        v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v14 = (__int64 *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v10 == *v14 )
        {
LABEL_7:
          if ( v14 != (__int64 *)(v12 + 16 * v11) && v14[1] )
            v7 |= sub_18F6D00(v10, a2, a3, a4, a5);
        }
        else
        {
          v18 = 1;
          while ( v15 != -8 )
          {
            v19 = v18 + 1;
            v13 = (v11 - 1) & (v18 + v13);
            v14 = (__int64 *)(v12 + 16LL * v13);
            v15 = *v14;
            if ( v10 == *v14 )
              goto LABEL_7;
            v18 = v19;
          }
        }
      }
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( v5 != v8 );
    return v7;
  }
}
