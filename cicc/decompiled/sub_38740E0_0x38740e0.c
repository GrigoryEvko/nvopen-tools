// Function: sub_38740E0
// Address: 0x38740e0
//
__int64 __fastcall sub_38740E0(__int64 a1, __int64 a2)
{
  bool v3; // zf
  unsigned int v4; // r8d
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 *v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r12
  _QWORD *v11; // rdx
  unsigned int v12; // esi
  int v13; // eax
  int v14; // eax
  int v15; // esi
  __int64 *v16; // r10
  int v17; // edx
  int v18; // r11d
  int v19; // eax
  __int64 v20; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v21[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_DWORD *)(a1 + 180) == *(_DWORD *)(a1 + 184);
  v20 = a2;
  if ( !v3 )
  {
    v4 = *(_DWORD *)(a1 + 112);
    v5 = a1 + 88;
    if ( v4 )
    {
      v6 = *(_QWORD *)(a1 + 96);
      result = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v6 + 8 * result);
      v9 = *v8;
      if ( a2 == *v8 )
        return result;
      v18 = 1;
      v16 = 0;
      while ( v9 != -8 )
      {
        if ( v16 || v9 != -16 )
          v8 = v16;
        result = (v4 - 1) & (v18 + (_DWORD)result);
        v9 = *(_QWORD *)(v6 + 8LL * (unsigned int)result);
        if ( a2 == v9 )
          return result;
        ++v18;
        v16 = v8;
        v8 = (__int64 *)(v6 + 8LL * (unsigned int)result);
      }
      v19 = *(_DWORD *)(a1 + 104);
      if ( !v16 )
        v16 = v8;
      ++*(_QWORD *)(a1 + 88);
      v17 = v19 + 1;
      if ( 4 * (v19 + 1) < 3 * v4 )
      {
        result = v4 - *(_DWORD *)(a1 + 108) - v17;
        if ( (unsigned int)result > v4 >> 3 )
          goto LABEL_14;
        v15 = v4;
LABEL_13:
        sub_3873F30(v5, v15);
        sub_3872C70(v5, &v20, v21);
        result = *(unsigned int *)(a1 + 104);
        v16 = (__int64 *)v21[0];
        a2 = v20;
        v17 = result + 1;
LABEL_14:
        *(_DWORD *)(a1 + 104) = v17;
        if ( *v16 != -8 )
          --*(_DWORD *)(a1 + 108);
        *v16 = a2;
        return result;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 88);
    }
    v15 = 2 * v4;
    goto LABEL_13;
  }
  v10 = a1 + 56;
  result = sub_3872C70(a1 + 56, &v20, v21);
  v11 = (_QWORD *)v21[0];
  if ( (_BYTE)result )
    return result;
  v12 = *(_DWORD *)(a1 + 80);
  v13 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  v14 = v13 + 1;
  if ( 4 * v14 >= 3 * v12 )
  {
    v12 *= 2;
    goto LABEL_25;
  }
  if ( v12 - *(_DWORD *)(a1 + 76) - v14 <= v12 >> 3 )
  {
LABEL_25:
    sub_3873F30(v10, v12);
    sub_3872C70(v10, &v20, v21);
    v11 = (_QWORD *)v21[0];
    v14 = *(_DWORD *)(a1 + 72) + 1;
  }
  *(_DWORD *)(a1 + 72) = v14;
  if ( *v11 != -8 )
    --*(_DWORD *)(a1 + 76);
  result = v20;
  *v11 = v20;
  return result;
}
