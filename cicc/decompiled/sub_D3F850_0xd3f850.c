// Function: sub_D3F850
// Address: 0xd3f850
//
__int64 __fastcall sub_D3F850(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r15
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 result; // rax
  unsigned int v12; // eax
  unsigned int v13; // r13d
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r15
  __int64 v20; // rdi
  __int64 v21; // rdi
  unsigned __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(a1 + 168);
  v9 = v8 + 48LL * *(unsigned int *)(a1 + 176);
  while ( v8 != v9 )
  {
    while ( 1 )
    {
      v9 -= 48;
      v10 = *(_QWORD *)(v9 + 16);
      if ( v10 == v9 + 32 )
        break;
      _libc_free(v10, a2);
      if ( v8 == v9 )
        goto LABEL_5;
    }
  }
LABEL_5:
  *(_DWORD *)(a1 + 176) = 0;
  if ( a3 )
    return sub_D3EC70(a1, a2);
  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v12 = 0;
    v13 = 0;
    while ( 1 )
    {
      if ( *(_DWORD *)(a1 + 180) <= v12 )
      {
        v19 = sub_C8D7D0(a1 + 168, a1 + 184, 0, 0x30u, v23, a6);
        v20 = v19 + 48LL * *(unsigned int *)(a1 + 176);
        if ( v20 )
          sub_D34480(v20, v13, a1);
        sub_D38990((__int64 *)(a1 + 168), v19, v15, v16, v17, v18);
        v21 = *(_QWORD *)(a1 + 168);
        result = v23[0];
        if ( a1 + 184 != v21 )
        {
          v22 = v23[0];
          _libc_free(v21, v19);
          result = v22;
        }
        ++*(_DWORD *)(a1 + 176);
        *(_QWORD *)(a1 + 168) = v19;
        *(_DWORD *)(a1 + 180) = result;
      }
      else
      {
        v14 = *(_QWORD *)(a1 + 168) + 48LL * v12;
        if ( v14 )
        {
          sub_D34480(v14, v13, a1);
          v12 = *(_DWORD *)(a1 + 176);
        }
        result = v12 + 1;
        *(_DWORD *)(a1 + 176) = result;
      }
      if ( ++v13 >= *(_DWORD *)(a1 + 16) )
        break;
      v12 = *(_DWORD *)(a1 + 176);
    }
  }
  return result;
}
