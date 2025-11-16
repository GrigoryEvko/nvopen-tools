// Function: sub_31B8EE0
// Address: 0x31b8ee0
//
__int64 __fastcall sub_31B8EE0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  char v4; // al
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // rdi
  int v12; // edx
  int v13; // r9d

  v2 = *a1;
  v3 = a1[1];
  while ( (unsigned __int8)sub_31B84B0(v2) != 1 && v2 != v3 )
    v2 = sub_318B4B0(v2);
  v4 = sub_31B84B0(v2);
  v5 = 0;
  if ( v4 )
  {
    v6 = *(unsigned int *)(a2 + 24);
    v7 = *(_QWORD *)(a2 + 8);
    if ( (_DWORD)v6 )
    {
      v8 = (v6 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v2 == *v9 )
      {
LABEL_8:
        if ( v9 != (__int64 *)(v7 + 16 * v6) )
          return v9[1];
      }
      else
      {
        v12 = 1;
        while ( v10 != -4096 )
        {
          v13 = v12 + 1;
          v8 = (v6 - 1) & (v12 + v8);
          v9 = (__int64 *)(v7 + 16LL * v8);
          v10 = *v9;
          if ( v2 == *v9 )
            goto LABEL_8;
          v12 = v13;
        }
      }
      return 0;
    }
  }
  return v5;
}
