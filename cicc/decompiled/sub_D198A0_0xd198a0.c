// Function: sub_D198A0
// Address: 0xd198a0
//
__int64 __fastcall sub_D198A0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 **v7; // rax
  unsigned __int8 **v8; // rdx
  int v10; // eax
  __int64 v11; // rsi
  int v12; // edx
  unsigned int v13; // eax
  unsigned __int8 *v14; // rcx
  int v15; // edi

  sub_D19710(a1, (__int64)a2, a3, a4, a5, a6);
  if ( *(_BYTE *)(a1 + 60) )
  {
    v7 = *(unsigned __int8 ***)(a1 + 40);
    v8 = &v7[*(unsigned int *)(a1 + 52)];
    if ( v7 != v8 )
    {
      while ( a2 != *v7 )
      {
        if ( v8 == ++v7 )
          goto LABEL_8;
      }
      return 0;
    }
  }
  else if ( sub_C8CA60(a1 + 32, (__int64)a2) )
  {
    return 0;
  }
LABEL_8:
  v10 = *(_DWORD *)(a1 + 344);
  v11 = *(_QWORD *)(a1 + 328);
  if ( v10 )
  {
    v12 = v10 - 1;
    v13 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = *(unsigned __int8 **)(v11 + 24LL * v13);
    if ( a2 != v14 )
    {
      v15 = 1;
      while ( v14 != (unsigned __int8 *)-4096LL )
      {
        v13 = v12 & (v15 + v13);
        v14 = *(unsigned __int8 **)(v11 + 24LL * v13);
        if ( a2 == v14 )
          return 0;
        ++v15;
      }
      return (unsigned int)sub_D14860(a2) ^ 1;
    }
    return 0;
  }
  return (unsigned int)sub_D14860(a2) ^ 1;
}
