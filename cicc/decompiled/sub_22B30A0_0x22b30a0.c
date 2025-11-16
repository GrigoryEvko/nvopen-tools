// Function: sub_22B30A0
// Address: 0x22b30a0
//
__int64 __fastcall sub_22B30A0(__int64 a1, __int64 *a2, __int64 **a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // r15d
  __int64 v9; // r14
  int v11; // r15d
  int v12; // eax
  int v13; // r8d
  __int64 *v14; // rcx
  unsigned int i; // edx
  __int64 v16; // rdi
  __int64 *v17; // rbx
  __int64 v18; // rsi
  char v19; // al
  unsigned int v20; // edx
  int v21; // [rsp+0h] [rbp-40h]
  unsigned int v22; // [rsp+4h] [rbp-3Ch]
  __int64 *v23; // [rsp+8h] [rbp-38h]

  v7 = *(_DWORD *)(a1 + 24);
  if ( v7 )
  {
    v9 = *(_QWORD *)(a1 + 8);
    v11 = v7 - 1;
    v12 = sub_22B2B70(*a2, (__int64)a2, (__int64)a3, a4, a5, a6);
    v13 = 1;
    v14 = 0;
    for ( i = v11 & v12; ; i = v11 & v20 )
    {
      v16 = *a2;
      v17 = (__int64 *)(v9 + 16LL * i);
      v18 = *v17;
      if ( (unsigned __int64)(*v17 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v16 - 1) > 0xFFFFFFFFFFFFFFFDLL )
      {
        if ( v18 == v16 )
        {
LABEL_13:
          *a3 = v17;
          return 1;
        }
      }
      else
      {
        v21 = v13;
        v22 = i;
        v23 = v14;
        v19 = sub_22AF4E0(v16, v18);
        v14 = v23;
        i = v22;
        v13 = v21;
        if ( v19 )
          goto LABEL_13;
        v18 = *v17;
      }
      if ( !v18 )
        break;
      if ( v18 == -1 && !v14 )
        v14 = v17;
      v20 = v13 + i;
      ++v13;
    }
    if ( !v14 )
      v14 = v17;
    *a3 = v14;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
