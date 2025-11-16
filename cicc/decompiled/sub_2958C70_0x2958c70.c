// Function: sub_2958C70
// Address: 0x2958c70
//
__int64 *__fastcall sub_2958C70(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdi
  int v5; // esi
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 *v9; // r15
  unsigned __int64 v10; // rax
  __int64 v11; // r12
  int v12; // ebx
  unsigned int i; // r14d
  __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  int v18; // eax
  int v19; // r8d
  __int64 *v21; // [rsp+18h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 8);
  if ( !v3 )
    return 0;
  v5 = v3 - 1;
  v6 = (v3 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( a1 != *v7 )
  {
    v18 = 1;
    while ( v8 != -4096 )
    {
      v19 = v18 + 1;
      v6 = v5 & (v18 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a1 == *v7 )
        goto LABEL_3;
      v18 = v19;
    }
    return 0;
  }
LABEL_3:
  v21 = (__int64 *)v7[1];
  if ( !v21 )
    return v21;
  v9 = (__int64 *)v7[1];
LABEL_5:
  while ( 2 )
  {
    v10 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1 + 48 != v10 )
    {
      if ( !v10 )
        BUG();
      v11 = v10 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 <= 0xA )
      {
        v12 = sub_B46E30(v11);
        if ( v12 )
        {
          for ( i = 0; v12 != i; ++i )
          {
            v14 = sub_B46EC0(v11, i);
            if ( *((_BYTE *)v9 + 84) )
            {
              v15 = (_QWORD *)v9[8];
              v16 = &v15[*((unsigned int *)v9 + 19)];
              if ( v15 == v16 )
                goto LABEL_19;
              while ( v14 != *v15 )
              {
                if ( v16 == ++v15 )
                  goto LABEL_19;
              }
            }
            else if ( !sub_C8CA60((__int64)(v9 + 7), v14) )
            {
LABEL_19:
              v21 = v9;
              v9 = (__int64 *)*v9;
              if ( !v9 )
                return v21;
              goto LABEL_5;
            }
          }
        }
      }
    }
    v9 = (__int64 *)*v9;
    if ( v9 )
      continue;
    return v21;
  }
}
