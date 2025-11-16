// Function: sub_1687E40
// Address: 0x1687e40
//
__int64 __fastcall sub_1687E40(__int64 a1, unsigned __int64 a2)
{
  __int16 v4; // ax
  __int64 v5; // r14
  _QWORD *v7; // rsi
  _DWORD *v8; // rdx
  __int64 v9; // rdi
  __int64 *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 i; // r14
  __int64 v16; // rdi
  _DWORD *v17; // r8
  _DWORD *v18; // r9
  __int64 v19; // rcx
  _DWORD *v20; // rax
  int v21; // edx
  _DWORD *v22; // rdx
  __int64 v23; // rax
  int v24; // [rsp+0h] [rbp-40h]
  unsigned int v25; // [rsp+8h] [rbp-38h]

  v4 = *(_WORD *)(a1 + 84) >> 4;
  if ( (_BYTE)v4 == 1 )
  {
    v25 = (a2 >> 11) ^ (a2 >> 8) ^ (a2 >> 5);
    v7 = (_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (*(_DWORD *)(a1 + 40) & v25));
    v8 = (_DWORD *)*v7;
    if ( *v7 )
    {
      v9 = *(_QWORD *)(a1 + 88);
      while ( 1 )
      {
        v11 = (unsigned int)v8[1];
        ++v8;
        if ( (_DWORD)v11 == -1 )
          break;
        v10 = (__int64 *)(v9 + 8 * v11);
        v5 = *v10;
        if ( a2 == *v10 )
        {
LABEL_20:
          --*(_QWORD *)(a1 + 48);
          v17 = 0;
          *(_DWORD *)(a1 + 56) ^= v25;
          v18 = (_DWORD *)*v7;
          v19 = ((__int64)v10 - v9) >> 3;
          v20 = (_DWORD *)*v7;
          do
          {
            while ( 1 )
            {
              v21 = v20[1];
              ++v20;
              if ( (_DWORD)v19 != v21 )
                break;
              v17 = v20;
            }
          }
          while ( v21 != -1 );
          *v17 = *(v20 - 1);
          *(v20 - 1) = -1;
          if ( v18[1] == -1 )
          {
            v24 = v19;
            sub_16856A0(v18);
            LODWORD(v19) = v24;
            v18 = 0;
          }
          *v7 = v18;
          *(_DWORD *)(*(_QWORD *)(a1 + 96) + 4LL * ((unsigned int)v19 >> 5)) ^= 1 << v19;
          return v5;
        }
      }
    }
    return 0;
  }
  if ( (_BYTE)v4 != 2 )
  {
    v5 = 0;
    if ( (_BYTE)v4 )
      return v5;
    if ( *(_QWORD *)(a1 + 32) )
      v25 = (*(__int64 (__fastcall **)(unsigned __int64))(a1 + 16))(a2);
    else
      v25 = (*(__int64 (__fastcall **)(unsigned __int64))a1)(a2);
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (*(_DWORD *)(a1 + 40) & v25));
    if ( v12 )
    {
      v13 = v12 + 4;
      v14 = *(unsigned int *)(v12 + 4);
      for ( i = *(_QWORD *)(a1 + 88); (_DWORD)v14 != -1; v13 += 4 )
      {
        v10 = (__int64 *)(i + 8 * v14);
        v16 = *v10;
        if ( *(_QWORD *)(a1 + 32) )
        {
          if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(a1 + 24))(v16, a2) )
            goto LABEL_19;
        }
        else if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(a1 + 8))(v16, a2) )
        {
LABEL_19:
          v5 = *v10;
          v9 = *(_QWORD *)(a1 + 88);
          v7 = (_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (*(_DWORD *)(a1 + 40) & v25));
          goto LABEL_20;
        }
        v14 = *(unsigned int *)(v13 + 4);
      }
    }
    return 0;
  }
  v25 = a2;
  v7 = (_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (*(_DWORD *)(a1 + 40) & (unsigned int)a2));
  v22 = (_DWORD *)*v7;
  if ( !*v7 )
    return 0;
  v9 = *(_QWORD *)(a1 + 88);
  while ( 1 )
  {
    v23 = (unsigned int)v22[1];
    ++v22;
    if ( (_DWORD)v23 == -1 )
      return 0;
    v10 = (__int64 *)(v9 + 8 * v23);
    v5 = *v10;
    if ( a2 == *v10 )
      goto LABEL_20;
  }
}
